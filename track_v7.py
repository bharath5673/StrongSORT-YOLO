import os
import sys
import argparse
import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
from pathlib import Path
from collections import defaultdict

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from random import random as ran
from datetime import datetime

import pandas as pd
from collections import Counter

import warnings
warnings.filterwarnings('ignore')


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # strongsort root directory
WEIGHTS = ROOT / 'weights'


if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov7 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT


def detect(opt):
    source, weights, show_vid, save_txt, imgsz, trace = opt.source, opt.yolo_weights, opt.show_vid, opt.save_txt, opt.img_size, opt.trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_crop = False
    strong_sort_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt'  # model.pt path
    save_txt = opt.save_txt  # save results to *.txt
    save_conf = opt.save_conf  # save confidences in --save-txt labels
    hide_labels = opt.hide_labels  # hide labels
    hide_conf = opt.hide_conf  # hide confidences
    hide_class = opt.hide_class  # hide IDs
    count = opt.count
    save_vid = opt.save_vid
    save_img = opt.save_img
    line_thickness = opt.line_thickness
    draw = opt.draw 
    # view_img = check_imshow() ### Find a fix to also run in Colab

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.exp_name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    # vid_path, vid_writer = None, None
    if webcam:
        # view_img = check_imshow() ### Find a fix to also run in Colab
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        nr_sources = 1
    # vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources
    to_check_vid_change, vid_writer = None, None

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA
            )
        )
    
    outputs = [None] * nr_sources

    trajectory = defaultdict(list)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    t0 = time.time()
    
    # Run tracking
    # model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

    # for path, img, im0s, vid_cap in dataset:
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        t1 = time_synchronized()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        t2 = time_synchronized()
        dt[0] += t2 - t1
        
        # Inference
        pred = model(img, augment=opt.augment)[0]
        t3 = time_synchronized()
        dt[1] += t3 - t2
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        dt[2] += time_synchronized() - t3

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            curr_frames[i] = im0
            p = Path(p)  # to Path
            # txt_file_name = p.name ### ????
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, .
            txt_path = f'{save_dir / "labels" / p.stem}.txt'  # im.txt

            s += '%gx%g ' % img.shape[2:]  # print string
            # imc = im0.copy() if save_crop else im0  # for save_crop ### ????
            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
            
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh ### ????
            if det.any():
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4]).type(torch.int)
                confs = det[:, 4]
                clss = det[:, 5].type(torch.uint8)

                # pass detections to strongsort
                t4 = time_synchronized()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_synchronized()
                dt[3] += t5 - t4
                
                # draw boxes for visualization
                if outputs[i].any():
                    for output in outputs[i]:
                        bboxes = output[0:4]
                        track_id, cls, conf = output[4:]

                        if draw:
                            # object trajectory
                            center = ((int(bboxes[0]) + int(bboxes[2])) // 2,(int(bboxes[1]) + int(bboxes[3])) // 2)
                            trajectory[track_id].append(center)
                            for c in range(-1, -21, -1): # Draw only the last 20 points
                                try:
                                    c1, c0 = trajectory[track_id][c], trajectory[track_id][c-1]
                                except IndexError:
                                    break
                                cv2.line(im0, c0, c1, colors[cls], line_thickness + 2)

                        if save_txt:
                            # frame_id, clas_id, track_id, xyxy bbox, detection conf
                            result_line = ' '.join(['%d'] * 7) %(frame_idx + 1, cls, track_id, *output[0:2], 
                                                                 *(output[[2, 3]] - output[[0, 1]] + 0.5))
                            result_line += f' {conf:.2e}\n' if save_conf else '\n' 
                            with open(txt_path, 'a') as f:
                                f.write(result_line)

                        if save_vid or save_crop or show_vid :  # Add bbox to image
                            label = None if hide_labels else (str(track_id) if hide_conf and hide_class else \
                                                              f'{track_id} {names[cls]}' if hide_conf else \
                                                              f'{track_id} {conf:.2f}' if hide_class else \
                                                              f'{track_id} {names[cls]} {conf:.2f}')
                            plot_one_box(bboxes, im0, label=label, color=colors[cls], line_thickness=line_thickness)

                ### Print time (inference + NMS)
                print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                print('No detections')

            if count:
                itemDict = {}
                ## NOTE: this works only if save-txt is true
                try:
                    df = pd.read_csv(txt_path, header=None, delim_whitespace=True)
                    df = df.iloc[:, 0:3]
                    df.columns = ["frame_id", "class", "track_id"]
                    df = df[['class','track_id']]
                    df = df.groupby('track_id')['class'].apply(list).apply(lambda x: sorted(x)).reset_index()

                    df.colums = ["track_id", "class"]
                    df['class'] = df['class'].apply(lambda x: Counter(x).most_common(1)[0][0])
                    vc = df['class'].value_counts()
                    vc = dict(vc)

                    vc2 = {}
                    for key, val in enumerate(names):
                        vc2[key] = val
                    itemDict = dict((vc2[key], value) for (key, value) in vc.items())
                    itemDict  = dict(sorted(itemDict.items(), key=lambda item: item[0]))

                except:
                    pass

                if save_txt:
                    ## overlay
                    display = im0.copy()
                    h, w = im0.shape[0], im0.shape[1]
                    x1, y1, x2, y2 = 10, 10, 10, 70
                    txt_size = cv2.getTextSize(str(itemDict), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.rectangle(im0, (x1, y1 + 1), (txt_size[0] * 2, y2),(0, 0, 0),-1)
                    cv2.putText(im0, '{}'.format(itemDict), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX,0.7, (210, 210, 210), 2)
                    cv2.addWeighted(im0, 0.7, display, 1 - 0.7, 0, im0)
            
            # current frame // tesing
            # cv2.imwrite('testing.jpg', im0)

            # Stream results
            # if show_vid:
            #     inf = (f'{s}Done. ({t2 - t1:.3f}s)')
            #     # cv2.putText(im0, str(inf), (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
            #     cv2.imshow(str(p), im0) ### Find a fix to also run in Colab
            #     if cv2.waitKey(1) == ord('q'):  # q to quit
            #         break

            # Save results (image with detections)
            if save_img and dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            if save_vid and dataset.mode == 'video':
                if to_check_vid_change != save_path:
                    to_check_vid_change = save_path
                    if vid_writer is not None:
                        vid_writer.release()  # release previous video writer
                    # if vid_cap is not None:  # video
                    #     fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    #     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    #     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    # else:  # stream
                    #     fps, w, h = 30, im0.shape[1], im0.shape[0]
                    #     save_path += '.mp4'
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fourcc = int(vid_cap.get(cv2.CAP_PROP_FOURCC))
                    vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                vid_writer.write(im0)

            prev_frames[i] = curr_frames[i]

    if save_txt or save_vid or save_img:
        print(f'Results saved to {save_dir}')
    print(f'All done in {time.time() - t0:.3f}s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default='weights/yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true',default=True, help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-img', action='store_true', help='save results to *.jpg')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',default=True, help='do not save images/videos')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/track', help='save results to project/name')
    parser.add_argument('--exp-name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--count', action='store_true', help='display all MOT counts results on screen')
    parser.add_argument('--draw', action='store_true', help='display object trajectory lines')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            detect(opt)
            strip_optimizer(opt.weights)
        else:
            detect(opt)
