import os
import glob
import pandas as pd
from collections import Counter
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

total = []
for text in glob.glob(str('runs/track/exp67/labels/*.txt')):
    name = Path(text).stem

    current_dir=os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(current_dir, text), header=None, delim_whitespace=True)
    df = df.iloc[:,0:8]
    df.columns=["frameid" ,"class","trackid", "side", "xmin", "ymax","w","h"]
    

    left_df = df[df['side'] == 0]   ## LEFT
    right_df = df[df['side'] == 1]  ## RIGHT

    sides = left_df, right_df

    result=[]

    for i, df in enumerate(sides):

        df = df[['class','trackid']]
        df = (df.groupby('trackid')['class'].apply(list)
                  .apply(lambda x:sorted(x))).reset_index()
        df.colums = ["trackid","class"]

        df['class']=df['class'].apply(lambda x: Counter(x).most_common(1)[0][0])
        vc = df['class'].value_counts()
        vc = dict(vc)

        vc2 = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', \
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', \
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', \
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', \
        38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', \
        48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', \
        58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', \
        68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',\
        78: 'hair drier', 79: 'toothbrush'}

        vc = dict((vc2[key], value) for (key, value) in vc.items())

        side = str
        if i == 0:
            side = 'left_side'
        if i == 1:
            side = 'right_side'

        val = {side : vc}

        total.append({name : val})

        lst = {name : val}
        print(lst)

        with open('COUNTS_LR.txt', 'a') as f:
            f.write(str(lst)+'\n') 

csv = pd.DataFrame(total) 
# csv.to_csv('report.csv') 