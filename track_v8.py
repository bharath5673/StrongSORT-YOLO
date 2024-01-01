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
from multiprocessing import Pool

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
model.overrides['conf'] = 0.3  # NMS confidence threshold
model.overrides['iou'] = 0.4  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image
# model.overrides['classes'] = 2 ## define classes
names = model.names
names = {value: key for key, value in names.items()}
colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')


tracking_trajectories = {}
def process(image, track=True):
    global input_video_name
    bboxes = []
    frameId = 0
    # Place this code outside the loop to avoid creating the file multiple times
    if not os.path.exists('output'):
        os.makedirs('output')
    labels_file_path = os.path.abspath(f'./output/{input_video_name}_labels.txt')

    # Open the file in 'a' (append) mode
    with open(labels_file_path, 'a') as file:
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

                        color_ = [int(c) for c in colors[int(classes)]]
                        # cv2.fillPoly(image, [np.int32(polygon)], color_) 
                        mask = image.copy()
                        cv2.fillPoly(mask, [np.int32(polygon)], color_) 
                        alpha = 0.5  # Adjust the transparency level
                        blended_image = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)
                        image = blended_image.copy()


        for item in bboxes:
            bbox_coords, scores, classes, *id_ = item if len(item) == 4 else (*item, None)
            line = f'{frameId} {int(classes)} {int(id_[0])} {round(float(scores), 3)} {int(bbox_coords[0])} {int(bbox_coords[1])} {int(bbox_coords[2])} {int(bbox_coords[3])} -1 -1 -1 -1\n'
            # print(line)
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

                    color_ = [int(c) for c in colors[int(classes)]]
                    # cv2.fillPoly(image, [np.int32(polygon)], color_) 
                    mask = image.copy()
                    cv2.fillPoly(mask, [np.int32(polygon)], color_) 
                    alpha = 0.5  # Adjust the transparency level
                    blended_image = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)
                    image = blended_image.copy()


    return image



def process_video(args):
    print(args)
    source = args['source']
    track_ = args['track']
    count_ = args['count']


    global input_video_name
    cap = cv2.VideoCapture(int(source) if source == '0' else source)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec if needed (e.g., 'XVID')
    # input_video_name = source.split('.')[0]  # Get the input video name without extension
    input_video_name = os.path.splitext(os.path.basename(source))[0]
    # print('testing : ', input_video_name)
    out = cv2.VideoWriter(f'output/{input_video_name}_output.mp4', fourcc, 15, (frame_width, frame_height))

    if not cap.isOpened():
        print(f"Error: Could not open video file {source}.")
        return

    frameId = 0
    start_time = time.time()
    fps_str = str()

    while True:
        frameId += 1
        ret, frame = cap.read()
        frame1 = frame.copy()
        if not ret:
            break

        frame = process(frame1, track_)

        if not track_ and count_:
            print('[INFO] count works only when objects are tracking.. so use: --track --count')
            break

        if track_ and count_:
            itemDict={}
            ## NOTE: this works only if save-txt is true
            try:
                df = pd.read_csv('output/'+input_video_name+'_labels.txt' , header=None, delim_whitespace=True)
                # print(df)
                df = df.iloc[:,0:3]
                df.columns=["frameid" ,"class","trackid"]
                df = df[['class','trackid']]
                df = (df.groupby('trackid')['class']
                          .apply(list)
                          .apply(lambda x:sorted(x))
                         ).reset_index()
                df['class']=df['class'].apply(lambda x: Counter(x).most_common(1)[0][0])
                vc = df['class'].value_counts()
                vc = dict(vc)

                vc2 = {}
                for key, val in enumerate(names):
                    vc2[key] = val
                itemDict = dict((vc2[key], value) for (key, value) in vc.items())
                itemDict  = dict(sorted(itemDict.items(), key=lambda item: item[0]))
                # print(itemDict)

            except:
                pass

            ## overlay
            display = frame.copy()
            h, w = frame.shape[0], frame.shape[1]
            x1, y1, x2, y2 =10, 10, 10, 70
            txt_size = cv2.getTextSize(str(itemDict), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(frame, (x1, y1 + 1), (txt_size[0] * 2, y2),(0, 0, 0),-1)
            cv2.putText(frame, '{}'.format(itemDict), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX,0.7, (210, 210, 210), 2)
            cv2.addWeighted(frame, 0.7, display, 1 - 0.7, 0, frame)

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if frameId % 10 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps_current = 10 / elapsed_time  # Calculate FPS over the last 20 frames
            fps_str = f'FPS: {fps_current:.2f}'
            start_time = time.time()  # Reset start_time for the next 20 frames

        cv2.putText(frame, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(f"yolo_{source}", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process video with YOLO.')
    parser.add_argument('--source', nargs='+', type=str, default='0', help='Input video file paths or camera indices')
    parser.add_argument('--track', action='store_true', help='if track objects')
    parser.add_argument('--count', action='store_true', help='if count objects')

    args = parser.parse_args()

    # Create a list of dictionaries containing the arguments for each process
    process_args_list = [{'source': source, 'track': args.track, 'count': args.count} for source in args.source]

    with Pool(processes=len(process_args_list)) as pool:
        pool.map(process_video, process_args_list)
