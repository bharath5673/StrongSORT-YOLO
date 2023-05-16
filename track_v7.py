import os
import sys
import argparse

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
from pathlib import Path

import cv2
import torch
# import torch.backends.cudnn as cudnn
import numpy as np

# import pandas as pd
# from collections import Counter

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
from yolov7.utils.plots import get_rgb_colors
from yolov7.utils.datasets import LoadImages
from yolov7.utils.general import (check_img_size, check_requirements, check_imshow, 
                                  yolov5_non_max_suppression, apply_classifier, scale_coords, 
                                  xyxy2xywh, strip_optimizer, set_logging, increment_path,
                                  save_argparser_arguments)
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT



def _build_strong_sort(opt):
    return StrongSORT(
        device=select_device(opt.device), 
        max_appearance_distance=opt.appearance_gate, 
        nn_budget=opt.feature_bank_size, 
        max_iou_distance=opt.iou_gate, 
        max_age=opt.max_age, 
        n_init=opt.init_period, 
        ema_alpha=opt.feature_momentum, 
        mc_lambda=opt.appearance_lambda, 
        matching_cascade=opt.matching_cascade,
        only_position=opt.motion_only_position,
        motion_gate_coefficient=opt.motion_gate_coefficient,
        max_centroid_distance=opt.max_centroid_distance
    )


def detect(opt):
    assert os.path.isdir(opt.source) or os.path.isfile(opt.source), 'Source must be a video file or a directory'
    # strong_sort_weights = opt.strong_sort_weights  # re-id model.pt path
    # view_img = check_imshow()  # Cannot run in Colab

    # Directories
    save_dir = Path(
        increment_path(Path(opt.project).absolute() / opt.name, exist_ok=opt.exist_ok))  # increment path
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make labels dir

    if opt.save_txt:
        save_txt_fmt = ' '.join(['%d'] * 7 + (['%.2f'] if opt.save_conf else []))
        save_txt_cols = list(range(8 if opt.save_conf else 7))

    save_argparser_arguments(opt, str(save_dir / 'arguments.txt'), opt.exist_ok)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt.yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(opt.img_size, s=stride)  # check img_size

    if opt.trace:
        model = TracedModel(model, device, imgsz)

    if half:
        model.half()  # to FP16

    # # Second-stage classifier
    # classify = False
    # if classify:
    #     modelc = load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    dataset = LoadImages(opt.source, img_size=imgsz, stride=stride)
    
    if opt.save_img and not dataset.video_flag[0]:
        (save_dir / 'images' / 'imgs').mkdir(parents=True, exist_ok=True)

    # cfg = get_config(config_file=opt.config_strongsort)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = get_rgb_colors(len(names), cmin=50, cmax=200, gray_colors=False)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    t0 = time.time()
    
    # model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup

    # Run tracking
    dt = [0.0, 0.0, 0.0, 0.0]
    strong_sort, actual_path, vid_writer, curr_frames, prev_frames, txt_path = [None] * 6
    for path, img, im0, vid_cap in dataset:
        curr_frames = im0
        path_base_name = os.path.splitext(os.path.basename(path))[0]
        
        if actual_path != path:
            if dataset.mode == 'video':
                if not opt.video_sequence or strong_sort is None:
                    strong_sort = _build_strong_sort(opt)
                    trajectorys = {}
                if opt.save_img:
                    (save_dir / 'images' / path_base_name).mkdir(parents=True, exist_ok=True)
                if opt.save_txt:
                    txt_path = str(save_dir / 'labels' / f'{path_base_name}.txt')
                if opt.save_vid:
                    try:
                        vid_writer.release()
                    except AttributeError:
                        pass
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fourcc = int(vid_cap.get(cv2.CAP_PROP_FOURCC))
                    vid_writer = cv2.VideoWriter(
                        str(save_dir / os.path.basename(path)), fourcc, fps, (w, h))
            else:
                if strong_sort is None:
                    strong_sort = _build_strong_sort(opt)
                    trajectorys = {}
                if opt.save_vid and vid_writer is None:
                    vid_writer = cv2.VideoWriter(str(save_dir / 'video_from_imgs.mp4'), 
                                                 cv2.VideoWriter_fourcc(*'mp4v'), 
                                                 20, im0.shape[-2:][::-1])
                if opt.save_txt and txt_path is None:
                    txt_path = str(save_dir / 'labels' / 'image_sequence.txt')
            actual_path = path

        t1 = time_synchronized()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0
        t2 = time_synchronized()
        dt[0] += t2 - t1
        
        # Inference
        pred = model(img, augment=opt.augment)[0]
        t3 = time_synchronized()
        dt[1] += t3 - t2
        # Apply NMS
        pred = yolov5_non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        dt[2] += time_synchronized() - t3

        # # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0)
        
        # Process detections
        detections = pred[0] # Only one batch was forward in the detection
        frame_id = getattr(dataset, 'frame', dataset.count)
        total_frames = getattr(dataset, 'nframes', dataset.nf - sum(dataset.video_flag))

        result_message = 'source %d/%d (%dx%d %s) | frame %d/%d |' %(
            dataset.count + 1, dataset.nf, *img.shape[2:][::-1], 
            dataset.mode, frame_id, total_frames)
        
        if opt.ecc:  # camera motion compensation
            strong_sort.tracker.camera_update(prev_frames, curr_frames)
        
        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh ### ????
        if detections.any():
            # Rescale boxes from img_size to im0 size
            detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], im0.shape).round()

            xyxys = detections[:, :4].type(torch.int32)
            confs = detections[:, 4]
            classes = detections[:, 5].type(torch.int32)

            cls_counts = zip(*torch.unique(classes, return_counts=True))
            cls_counts = [f'{names[i.item()]} x{j.item()}' for i, j in cls_counts]
            result_message += f' {" ".join(cls_counts)} |'

            # pass detections to strongsort
            t4 = time_synchronized()
            sort_output = strong_sort.update(xyxys.cpu(), confs.cpu(), classes.cpu(), im0)
            t5 = time_synchronized()
            dt[3] += t5 - t4
            
            if sort_output.any():
                sort_output = np.c_[np.repeat([[frame_id]], sort_output.shape[0], axis=0), sort_output]
                if opt.save_txt:
                    # frame_id, track_id, clas_id, tlwh bbox, detection conf
                    with open(txt_path, mode='a') as f:
                        np.savetxt(f, sort_output[:, save_txt_cols], fmt=save_txt_fmt)
                
                for output in sort_output:
                    track_id, cls = output[[1, 2]].astype(np.int32) 
                    tlwh = output[3:7]
                    conf = output[7]
                    if opt.draw_trajectory:
                        center = tuple((tlwh[:2] + tlwh[2:] / 2).round().astype(np.int32).tolist())
                        track_trajs = trajectorys.setdefault(track_id, [])
                        track_trajs.append(center)
                        for c in range(-1, -21, -1):  # Draw only the last 20 points
                            try:
                                c1, c0 = track_trajs[c], track_trajs[c-1]
                            except IndexError:
                                break
                            cv2.line(im0, c0, c1, colors[cls], 3)

                    if opt.save_vid:  # Add bbox to image
                        xyxy = tlwh.astype(np.int32)
                        xyxy[2:] = xyxy[:2] + xyxy[2:]
                        label = None if opt.hide_labels else (str(track_id) if opt.hide_conf and opt.hide_class else \
                                                              f'{track_id} {names[cls]}' if opt.hide_conf else \
                                                              f'{track_id} {conf:.2f}' if opt.hide_class else \
                                                              f'{track_id} {names[cls]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors[cls], line_thickness=opt.line_thickness)

        else:
            strong_sort.increment_ages()
            result_message += ' None detections |'

        if opt.verbose:
            print(f'{result_message} YOLO {dt[1]:.3e}s, StrongSORT {dt[3]:.3e}s')
        
        # if opt.count:
        #     itemDict = {}
        #     ## NOTE: this works only if save-txt is true
        #     try:
        #         df = pd.read_csv(txt_path, header=None, delim_whitespace=True)
        #         df = df.iloc[:, 0:3]
        #         df.columns = ["frame_id", "class", "track_id"]
        #         df = df[['class','track_id']]
        #         df = df.groupby('track_id')['class'].apply(list).apply(lambda x: sorted(x)).reset_index()

        #         df.colums = ["track_id", "class"]
        #         df['class'] = df['class'].apply(lambda x: Counter(x).most_common(1)[0][0])
        #         vc = df['class'].value_counts()
        #         vc = dict(vc)

        #         vc2 = {}
        #         for key, val in enumerate(names):
        #             vc2[key] = val
        #         itemDict = dict((vc2[key], value) for (key, value) in vc.items())
        #         itemDict  = dict(sorted(itemDict.items(), key=lambda item: item[0]))

        #     except:
        #         pass

        #     if opt.save_txt:
        #         ## overlay
        #         display = im0.copy()
        #         h, w = im0.shape[0], im0.shape[1]
        #         x1, y1, x2, y2 = 10, 10, 10, 70
        #         txt_size = cv2.getTextSize(str(itemDict), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        #         cv2.rectangle(im0, (x1, y1 + 1), (txt_size[0] * 2, y2),(0, 0, 0),-1)
        #         cv2.putText(im0, '{}'.format(itemDict), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX,0.7, (210, 210, 210), 2)
        #         cv2.addWeighted(im0, 0.7, display, 1 - 0.7, 0, im0)
        
        # current frame // tesing
        # cv2.imwrite('testing.jpg', im0)

        # Stream results
        # if show_vid:
        #     inf = (f'{result_message}Done. ({t2 - t1:.3f}s)')
        #     # cv2.putText(im0, str(inf), (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
        #     cv2.imshow(str(p), im0)  # Find a fix to also run in Colab
        #     if cv2.waitKey(1) == ord('q'):  # q to quit
        #         break

        if opt.save_img:
            padding = len(str(total_frames))
            file_name = f'{frame_id:0>{padding}}_{path_base_name}.jpg'
            if dataset.mode == 'image':
                save_img_result = cv2.imwrite(str(save_dir / 'images' / 'imgs' / file_name), im0)
            else:
                save_img_result = cv2.imwrite(str(save_dir / 'images' / path_base_name / file_name), im0)
            if not save_img_result and opt.verbose:
                print('Error while saving image/frame:')
                print('    - ID   :', frame_id)
                print('    - File :', path)
        
        if opt.save_vid:
            vid_writer.write(im0)

        prev_frames = curr_frames

    if opt.save_txt or opt.save_vid or opt.save_img:
        print(f'Results saved to {save_dir}')
    print(f'All done in {time.time() - t0:.3f}s')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    ### YOLO

    parser.add_argument(
        '--yolo-weights', 
        nargs='+', type=str, default='weights/yolov7-tiny.pt', 
        help='path to yolo model file'
    )

    parser.add_argument(
        '--img-size', 
        type=int, default=640, 
        help='inference size (pixels) of yolo detector'
    )

    parser.add_argument(
        '--conf-thres', 
        type=float, default=0.5, 
        help='object detection confidence threshold'
    )
    
    parser.add_argument(
        '--iou-thres', 
        type=float, default=0.5, 
        help='IoU threshold for NMS'
    )

    parser.add_argument(
        '--classes', 
        nargs='+', type=int, 
        help='filter detections by class index, e.g.: --class 0 2 3'
    )

    parser.add_argument(
        '--agnostic-nms', 
        action='store_true', 
        help='class-agnostic NMS'
    )

    parser.add_argument(
        '--augment', 
        action='store_true', 
        help='augmented inference'
    )

    parser.add_argument(
        '--update', 
        action='store_true', 
        help='update all models'
    )

    parser.add_argument(
        '--trace', 
        action='store_true', 
        help='trace model'
    )

    ### General

    parser.add_argument(
        '--source', 
        type=str, default='.', 
        help='source data (video file or directory with images or/and videos) for tracking'
    )

    parser.add_argument(
        '--video-sequence', 
        action='store_true', 
        help='keep tracker alive (do not reinstantiate) between videos'
    )

    parser.add_argument(
        '--device', 
        default='cpu', 
        help='cuda device, i.e. 0 or 0,1,2,3 or cpu'
    )

    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='report tracking informations'
    )

    ### Saving results

    parser.add_argument(
        '--project', 
        default='runs/track', 
        help='save results to /project/name'
    )

    parser.add_argument(
        '--name', 
        default='exp', 
        help='save results to /project/name'
    )

    parser.add_argument(
        '--exist-ok', 
        action='store_true', 
        help='existing /project/name ok, do not increment dir path'
    )

    parser.add_argument(
        '--save-txt', 
        action='store_true', 
        help='save tracking results to /project/name/labels/*.txt'
    )

    parser.add_argument(
        '--save-img', 
        action='store_true', 
        help='save processed images/frames to /project/name/images/*/*.jpg'
    )

    parser.add_argument(
        '--save-conf', 
        action='store_true', 
        help='save detection confidences in --save-txt results'
    )

    parser.add_argument(
        '--save-vid', 
        action='store_true', 
        help='save videos with the tracking results'
    )

    ### Visualization

    # parser.add_argument(
    #     '--show-vid', 
    #     action='store_true', 
    #     help='display results while processing (not supported for now)'
    # )

    # parser.add_argument(
    #     '--count', action='store_true', 
    #     help='display all MOT counts results on screen (not supported for now)'
    # )

    parser.add_argument(
        '--line-thickness', 
        type=int, default=2, 
        help='bounding box thickness in pixels'
    )

    parser.add_argument(
        '--hide-labels', 
        action='store_true', 
        help='only draw bounding box rectangle'
    )

    parser.add_argument(
        '--hide-conf', 
        action='store_true', 
        help='hide detection confidence in bounding box label'
    )

    parser.add_argument(
        '--hide-class', 
        action='store_true', 
        help='hide class id in bounding box label'
    )

    parser.add_argument(
        '--draw-trajectory', 
        action='store_true', 
        help='display object trajectory lines'
    )

    ### StrongSORT

    ### Now need to change the model in the source code
    ### See https://github.com/KaiyangZhou/deep-person-reid for more models
    # parser.add_argument(
    #     '--strong-sort-weights', 
    #     type=str, default=(WEIGHTS / 'osnet_x0_25_msmt17.pt'),
    #     help='path to feature extractor model file for peron/object reid'
    # )

    ### Now need to supply configurations as arguments to parser
    # parser.add_argument(
    #     '--config-strongsort', 
    #     type=str, default='strong_sort/configs/strong_sort.yaml', 
    #     help='path to yaml file for StrongSORT() instantiation/configuration'
    # )

    parser.add_argument(
        '--matching-cascade',
        action='store_true',
        help='apply DeepSORT matching cascade'
    )
    
    parser.add_argument(
        '--appearance-lambda',  # Old strong_sort.yaml STRONGSORT.MC_LAMBDA
        type=float, default=0.995,
        help='appearance cost weight for appearance-motion cost matrix calculation'
    )
    
    parser.add_argument(
        '--iou-gate',  # Old strong_sort.yaml STRONGSORT.MAX_IOU_DISTANCE
        type=float, default=0.7,
        help='IoU distance gate for the final IoU matching'
    )

    parser.add_argument(
        '--ecc',  # Old strong_sort.yaml STRONGSORT.ECC
        action='store_true',
        help='apply camera motion compensation using ECC'
    )
    
    ### TODO: choose between feature update or feature bank
    parser.add_argument(
        '--feature-bank-size',  # Old strong_sort.yaml STRONGSORT.NN_BUDGET
        type=int, default=100,
        help='num of features to store per Track for appearance distance calculation'
    )

    parser.add_argument(
        '--init-period',  # Old strong_sort.yaml STRONGSORT.N_INIT
        type=int, default=3,
        help='size of Track initialization period in frames'
    )
    
    parser.add_argument(
        '--max-age',  # Old strong_sort.yaml STRONGSORT.MAX_AGE
        type=int, default=10,
        help='max period which a Track survive without assignments in frames'
    )

    ### Irrelevant at the moment since the appearance cost matrix in calculated
    ### using the NearestNeighborDistanceMetric feature bank. The original
    ### StrongSORT implementation don't use the feature bank but the feature
    ### stored inside Track object and updated using the momentum term
    parser.add_argument(
        '--feature-momentum',  # Old strong_sort.yaml STRONGSORT.EMA_ALPHA
        type=float, default=0.9,
        help='momentum term for Track.feature update'
    )
    
    parser.add_argument(
        '--appearance-gate',  # Old strong_sort.yaml STRONGSORT.MAX_DIST
        type=float, default=0.2,
        help='track-detection associations with appearance cost greater than this value are disregarded'
    )

    parser.add_argument(
        '--motion-only-position',
        action='store_true', 
        help='use only centroid position to compute motion cost'
    )

    parser.add_argument(
        '--motion-gate-coefficient',
        type=float, default=1.0,
        help='coefficient that multiplies the motion gate to control track-detection associations'
    )

    parser.add_argument(
        '--max-centroid-distance',
        type=int, default=None,
        help='max distance in pixels between track and detection centroids for track-detection match'
    )

    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            detect(opt)
            strip_optimizer(opt.yolo_weights)
        else:
            detect(opt)
