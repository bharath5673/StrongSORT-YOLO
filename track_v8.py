## conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
## pip install ultralytics

import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import Counter, deque
import pandas as pd
import argparse
from multiprocessing import Process

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
model.overrides['conf'] = 0.3  # NMS confidence threshold
model.overrides['iou'] = 0.4  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image
# model.overrides['classes'] = 2 ## define classes
names = model.names
names = {value: key for key, value in names.items()}



tracking_trajectories = {}
def process(image, track=True):
    global input_video_name
    bboxes = []
    # Place this code outside the loop to avoid creating the file multiple times
    if not os.path.exists('output'):
        os.makedirs('output')

    # Open the file in 'a' (append) mode
    with open('output/'+input_video_name+'_labels.txt', 'a') as file:
        if track is True:
            results = model.track(image, verbose=False, device=0, persist=True, tracker="botsort.yaml")

            for id_ in list(tracking_trajectories.keys()):
                if id_ not in [int(bbox.id) for predictions in results if predictions is not None for bbox in predictions.boxes if bbox.id is not None]:
                    del tracking_trajectories[id_]

            for predictions in results:
                if predictions is None:
                    continue

                if predictions.boxes is None or predictions.masks is None or predictions.boxes.id is None:
                    continue

                for bbox, masks in zip(predictions.boxes, predictions.masks):
                    for scores, classes, bbox_coords, id_ in zip(bbox.conf, bbox.cls, bbox.xyxy, bbox.id):
                        xmin    = bbox_coords[0]
                        ymin    = bbox_coords[1]
                        xmax    = bbox_coords[2]
                        ymax    = bbox_coords[3]
                        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,225), 2)
                        bboxes.append([bbox_coords, scores, classes, id_])

                        label = (' '+f'ID: {int(id_)}'+' '+str(predictions.names[int(classes)]) + ' ' + str(round(float(scores) * 100, 1)) + '%')
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                        dim, baseline = text_size[0], text_size[1]
                        cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] //3) - 20, int(ymin) - dim[1] + baseline), (30,30,30), cv2.FILLED)
                        cv2.putText(image,label,(int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                        centroid_x = (xmin + xmax) / 2
                        centroid_y = (ymin + ymax) / 2

                        # Append centroid to tracking_points
                        if id_ is not None and int(id_) not in tracking_trajectories:
                            tracking_trajectories[int(id_)] = deque(maxlen=5)
                        if id_ is not None:
                            tracking_trajectories[int(id_)].append((centroid_x, centroid_y))

                    # Draw trajectories
                    for id_, trajectory in tracking_trajectories.items():
                        for i in range(1, len(trajectory)):
                            cv2.line(image, (int(trajectory[i-1][0]), int(trajectory[i-1][1])), (int(trajectory[i][0]), int(trajectory[i][1])), (255, 255, 255), 2)

                    for mask in masks.xy:
                        polygon = mask
                        cv2.polylines(image, [np.int32(polygon)], True, (255, 0, 0), thickness=2)

        for item in bboxes:
            bbox_coords, scores, classes, *id_ = item if len(item) == 4 else (*item, None)
            line = f'{frameId} {int(classes)} {int(id_[0])} {round(float(scores), 3)} {int(bbox_coords[0])} {int(bbox_coords[1])} {int(bbox_coords[2])} {int(bbox_coords[3])} -1 -1 -1 -1\n'
            file.write(line)

    if not track:
        results = model.predict(image, verbose=False, device=0)  # predict on an image
        for predictions in results:
            if predictions is None:
                continue  # Skip this image if YOLO fails to detect any objects
            if predictions.boxes is None or predictions.masks is None:
                continue  # Skip this image if there are no boxes or masks

            for bbox, masks in zip(predictions.boxes, predictions.masks):              
                for scores, classes, bbox_coords in zip(bbox.conf, bbox.cls, bbox.xyxy):
                    xmin    = bbox_coords[0]
                    ymin    = bbox_coords[1]
                    xmax    = bbox_coords[2]
                    ymax    = bbox_coords[3]
                    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,225), 2)
                    bboxes.append([bbox_coords, scores, classes])

                    label = (' '+str(predictions.names[int(classes)]) + ' ' + str(round(float(scores) * 100, 1)) + '%')
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
                    dim, baseline = text_size[0], text_size[1]
                    cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] //3) - 20, int(ymin) - dim[1] + baseline), (30,30,30), cv2.FILLED)
                    cv2.putText(image,label,(int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                for mask in masks.xy:
                    polygon = mask
                    cv2.polylines(image, [np.int32(polygon)], True, (255, 0, 0), thickness=2)

    return image



def process_video(input_path, output_path, track, count):
    global input_video_name
    cap = cv2.VideoCapture(int(input_path) if input_path == '0' else input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    input_video_name = input_path.split('.')[0]
    out = cv2.VideoWriter('output/'+output_path, fourcc, fps, (frame_width, frame_height))

    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_path}")
        return

    frameId = 0
    start_time = time.time()
    fps_str = ""

    while True:
        frameId += 1
        ret, frame = cap.read()
        frame1 = frame.copy()
        if not ret:
            break

        frame = process(frame1, track)

        if not track and count:
            print('[INFO] count works only when objects are tracking.. so use: --track --count')
            break

        if track and count:
            item_dict = {}
            try:
                df = pd.read_csv(f'output/{input_video_name}_labels.txt', header=None, delim_whitespace=True)
                df = df.iloc[:, 0:3]
                df.columns = ["frameid", "class", "trackid"]
                df = df[['class', 'trackid']]
                df = (df.groupby('trackid')['class']
                        .apply(list)
                        .apply(lambda x: sorted(x))
                        ).reset_index()
                df['class'] = df['class'].apply(lambda x: Counter(x).most_common(1)[0][0])
                vc = df['class'].value_counts()
                vc = dict(vc)

                vc2 = {}
                for key, val in enumerate(names):
                    vc2[key] = val
                item_dict = dict((vc2[key], value) for (key, value) in vc.items())
                item_dict = dict(sorted(item_dict.items(), key=lambda item: item[0]))
            except:
                pass

            display = frame.copy()
            h, w = frame.shape[0], frame.shape[1]
            x1, y1, x2, y2 = 10, 10, 10, 70
            txt_size = cv2.getTextSize(str(item_dict), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(frame, (x1, y1 + 1), (txt_size[0] * 2, y2), (0, 0, 0), -1)
            cv2.putText(frame, '{}'.format(item_dict), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 2)
            cv2.addWeighted(frame, 0.7, display, 1 - 0.7, 0, frame)

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if frameId % 10 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps_current = 10 / elapsed_time
            fps_str = f'FPS: {fps_current:.2f}'
            start_time = time.time()

        cv2.putText(frame, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("yolo", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process multiple videos with YOLO.')
    parser.add_argument('--sources', nargs='+', required=True, help='Input video file paths or camera indices')
    parser.add_argument('--track', action='store_true', help='if track objects')
    parser.add_argument('--count', action='store_true', help='if count objects')

    args = parser.parse_args()

    if len(args.sources) == 0:
        print("Error: Please provide at least one source.")
        exit()

    if args.track and not args.count:
        print("Warning: Tracking is enabled, but counting is not. Counting works when tracking is enabled.")

    processes = []

    for source in args.sources:
        output_video_name = source.split('.')[0] + '_output.mp4'
        p = Process(target=process_video, args=(source, output_video_name, args.track, args.count))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


