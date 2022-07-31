import argparse
import yaml
import math
import os
import struct
import torch
from utils.torch_utils import select_device


class YoloLayers():
    def get_route(self, n, layers):
        route = 0
        for i, layer in enumerate(layers):
            if i <= n:
                route += layer[1]
            else:
                break
        return route

    def route(self, layers=""):
        return "\n[route]\n" + \
               "layers=%s\n" % layers

    def reorg(self):
        return "\n[reorg]\n"

    def shortcut(self, route=-1, activation="linear"):
        return "\n[shortcut]\n" + \
               "from=%d\n" % route + \
               "activation=%s\n" % activation

    def maxpool(self, stride=1, size=1):
        return "\n[maxpool]\n" + \
               "stride=%d\n" % stride + \
               "size=%d\n" % size

    def upsample(self, stride=1):
        return "\n[upsample]\n" + \
               "stride=%d\n" % stride

    def convolutional(self, bn=False, size=1, stride=1, pad=1, filters=1, groups=1, activation="linear"):
        b = "batch_normalize=1\n" if bn is True else ""
        g = "groups=%d\n" % groups if groups > 1 else ""
        return "\n[convolutional]\n" + \
               b + \
               "filters=%d\n" % filters + \
               "size=%d\n" % size + \
               "stride=%d\n" % stride + \
               "pad=%d\n" % pad + \
               g + \
               "activation=%s\n" % activation

    def yolo(self, mask="", anchors="", classes=80, num=3):
        return "\n[yolo]\n" + \
               "mask=%s\n" % mask + \
               "anchors=%s\n" % anchors + \
               "classes=%d\n" % classes + \
               "num=%d\n" % num + \
               "scale_x_y=2.0\n" + \
               "beta_nms=0.6\n" + \
               "new_coords=1\n"


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch YOLOv5 conversion")
    parser.add_argument("-w", "--weights", required=True, help="Input weights (.pt) file path (required)")
    parser.add_argument("-c", "--yaml", help="Input cfg (.yaml) file path")
    parser.add_argument("-mw", "--width", help="Model width (default = 640 / 1280 [P6])")
    parser.add_argument("-mh", "--height", help="Model height (default = 640 / 1280 [P6])")
    parser.add_argument("-mc", "--channels", help="Model channels (default = 3)")
    parser.add_argument("--p6", action="store_true", help="P6 model")
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit("Invalid weights file")
    if not args.yaml:
        args.yaml = ""
    if not args.width:
        args.width = 1280 if args.p6 else 640
    if not args.height:
        args.height = 1280 if args.p6 else 640
    if not args.channels:
        args.channels = 3
    return args.weights, args.yaml, args.width, args.height, args.channels, args.p6


def get_width(x, gw, divisor=8):
    return int(math.ceil((x * gw) / divisor)) * divisor


def get_depth(x, gd):
    if x == 1:
        return 1
    r = int(round(x * gd))
    if x * gd - int(x * gd) == 0.5 and int(x * gd) % 2 == 0:
        r -= 1
    return max(r, 1)


pt_file, yaml_file, model_width, model_height, model_channels, p6 = parse_args()

model_name = pt_file.split(".pt")[0]
wts_file = model_name + ".wts" if "yolov7" in model_name else "yolov7_" + model_name + ".wts"
cfg_file = model_name + ".cfg" if "yolov7" in model_name else "yolov7_" + model_name + ".cfg"

if yaml_file == "":
    yaml_file = "models/" + model_name + ".yaml"
    if not os.path.isfile(yaml_file):
        yaml_file = "models/hub/" + model_name + ".yaml"
        if not os.path.isfile(yaml_file):
            raise SystemExit("YAML file not found")
elif not os.path.isfile(yaml_file):
    raise SystemExit("Invalid YAML file")

device = select_device("cpu")
model = torch.load(pt_file, map_location=device)["model"].float()

anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]
delattr(model.model[-1], "anchor_grid")
model.model[-1].register_buffer("anchor_grid", anchor_grid)

model.to(device).eval()

anchors = ""
masks = []

for k, v in model.state_dict().items():
    if "anchor_grid" in k:
        vr = v.cpu().numpy().tolist()
        a = v.reshape(-1).cpu().numpy().astype(float).tolist()
        anchors = str(a)[1:-1]
        num = 0
        for m in vr:
            mask = []
            for _ in range(len(m)):
                mask.append(num)
                num += 1
            masks.append(mask)

spp_idx = 0

with open(cfg_file, "w") as c:
    with open(yaml_file, "r", encoding="utf-8") as f:
        c.write("[net]\n")
        c.write("width=%d\n" % model_width)
        c.write("height=%d\n" % model_height)
        c.write("channels=%d\n" % model_channels)
        nc = 0
        depth_multiple = 0
        width_multiple = 0
        layers = []
        yoloLayers = YoloLayers()
        f = yaml.load(f, Loader=yaml.FullLoader)
        for topic in f:
            if topic == "nc":
                nc = f[topic]
            elif topic == "depth_multiple":
                depth_multiple = f[topic]
            elif topic == "width_multiple":
                width_multiple = f[topic]
            elif topic == "backbone" or topic == "head":
                for v in f[topic]:
                    if v[2] == "Focus":
                        layer = "\n# Focus\n"
                        blocks = 0
                        layer += yoloLayers.reorg()
                        blocks += 1
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple), size=v[3][1],
                                                          activation="silu")
                        blocks += 1
                        layers.append([layer, blocks])
                    if v[2] == "Conv":
                        layer = "\n# Conv\n"
                        blocks = 0
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple), size=v[3][1],
                                                          stride=v[3][2], activation="silu")
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "C3":
                        layer = "\n# C3\n"
                        blocks = 0
                        # SPLIT
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple) / 2,
                                                          activation="silu")
                        blocks += 1
                        layer += yoloLayers.route(layers="-2")
                        blocks += 1
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple) / 2,
                                                          activation="silu")
                        blocks += 1
                        # Residual Block
                        if len(v[3]) == 1 or v[3][1] is True:
                            for _ in range(get_depth(v[1], depth_multiple)):
                                layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple) / 2,
                                                                  activation="silu")
                                blocks += 1
                                layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple) / 2,
                                                                  size=3, activation="silu")
                                blocks += 1
                                layer += yoloLayers.shortcut(route=-3)
                                blocks += 1
                            # Merge
                            layer += yoloLayers.route(layers="-1, -%d" % (3 * get_depth(v[1], depth_multiple) + 3))
                            blocks += 1
                        else:
                            for _ in range(get_depth(v[1], depth_multiple)):
                                layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple) / 2,
                                                                  activation="silu")
                                blocks += 1
                                layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple) / 2,
                                                                  size=3, activation="silu")
                                blocks += 1
                            # Merge
                            layer += yoloLayers.route(layers="-1, -%d" % (2 * get_depth(v[1], depth_multiple) + 3))
                            blocks += 1
                        # Transition
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple),
                                                          activation="silu")
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "SPP":
                        spp_idx = len(layers)
                        layer = "\n# SPP\n"
                        blocks = 0
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple) / 2,
                                                          activation="silu")
                        blocks += 1
                        layer += yoloLayers.maxpool(size=v[3][1][0])
                        blocks += 1
                        layer += yoloLayers.route(layers="-2")
                        blocks += 1
                        layer += yoloLayers.maxpool(size=v[3][1][1])
                        blocks += 1
                        layer += yoloLayers.route(layers="-4")
                        blocks += 1
                        layer += yoloLayers.maxpool(size=v[3][1][2])
                        blocks += 1
                        layer += yoloLayers.route(layers="-6, -5, -3, -1")
                        blocks += 1
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple),
                                                          activation="silu")
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "SPPF":
                        spp_idx = len(layers)
                        layer = "\n# SPPF\n"
                        blocks = 0
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple) / 2,
                                                          activation="silu")
                        blocks += 1
                        layer += yoloLayers.maxpool(size=v[3][1])
                        blocks += 1
                        layer += yoloLayers.maxpool(size=v[3][1])
                        blocks += 1
                        layer += yoloLayers.maxpool(size=v[3][1])
                        blocks += 1
                        layer += yoloLayers.route(layers="-4, -3, -2, -1")
                        blocks += 1
                        layer += yoloLayers.convolutional(bn=True, filters=get_width(v[3][0], width_multiple),
                                                          activation="silu")
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "nn.Upsample":
                        layer = "\n# nn.Upsample\n"
                        blocks = 0
                        layer += yoloLayers.upsample(stride=v[3][1])
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "Concat":
                        route = v[0][1]
                        route = yoloLayers.get_route(route, layers) if route > 0 else \
                            yoloLayers.get_route(len(layers) + route, layers)
                        layer = "\n# Concat\n"
                        blocks = 0
                        layer += yoloLayers.route(layers="-1, %d" % (route - 1))
                        blocks += 1
                        layers.append([layer, blocks])
                    elif v[2] == "Detect":
                        for i, n in enumerate(v[0]):
                            route = yoloLayers.get_route(n, layers)
                            layer = "\n# Detect\n"
                            blocks = 0
                            layer += yoloLayers.route(layers="%d" % (route - 1))
                            blocks += 1
                            layer += yoloLayers.convolutional(filters=((nc + 5) * len(masks[i])), activation="logistic")
                            blocks += 1
                            layer += yoloLayers.yolo(mask=str(masks[i])[1:-1], anchors=anchors, classes=nc, num=num)
                            blocks += 1
                            layers.append([layer, blocks])
        for layer in layers:
            c.write(layer[0])

with open(wts_file, "w") as f:
    wts_write = ""
    conv_count = 0
    cv1 = ""
    cv3 = ""
    cv3_idx = 0
    for k, v in model.state_dict().items():
        if "num_batches_tracked" not in k and "anchors" not in k and "anchor_grid" not in k:
            vr = v.reshape(-1).cpu().numpy()
            idx = int(k.split(".")[1])
            if ".cv1." in k and ".m." not in k and idx != spp_idx:
                cv1 += "{} {} ".format(k, len(vr))
                for vv in vr:
                    cv1 += " "
                    cv1 += struct.pack(">f", float(vv)).hex()
                cv1 += "\n"
                conv_count += 1
            elif cv1 != "" and ".m." in k:
                wts_write += cv1
                cv1 = ""
            if ".cv3." in k:
                cv3 += "{} {} ".format(k, len(vr))
                for vv in vr:
                    cv3 += " "
                    cv3 += struct.pack(">f", float(vv)).hex()
                cv3 += "\n"
                cv3_idx = idx
                conv_count += 1
            elif cv3 != "" and cv3_idx != idx:
                wts_write += cv3
                cv3 = ""
                cv3_idx = 0
            if ".cv3." not in k and not (".cv1." in k and ".m." not in k and idx != spp_idx):
                wts_write += "{} {} ".format(k, len(vr))
                for vv in vr:
                    wts_write += " "
                    wts_write += struct.pack(">f", float(vv)).hex()
                wts_write += "\n"
                conv_count += 1
    f.write("{}\n".format(conv_count))
    f.write(wts_write)






# eflag@eflag-MS-7D43:~/Documents/yolov5/DeepStream-Yolo$ deepstream-app -c deepstream_app_config.txt 
# gstnvtracker: Loading low-level lib at /opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
# gstnvtracker: Batch processing is ON
# gstnvtracker: Past frame output is OFF
# [NvTrackerParams::getConfigRoot()] !!![WARNING] Empty config file path is provided. Will go ahead with default values
# [NvMultiObjectTracker] Initialized
# ERROR: ../nvdsinfer/nvdsinfer_model_builder.cpp:1484 Deserialize engine failed because file path: /home/eflag/Documents/yolov5/DeepStream-Yolo/model_b1_gpu0_fp32.engine open error
# 0:00:00.586940902 32181 0x563a8cd97120 WARN                 nvinfer gstnvinfer.cpp:635:gst_nvinfer_logger:<primary_gie> NvDsInferContext[UID 1]: Warning from NvDsInferContextImpl::deserializeEngineAndBackend() <nvdsinfer_context_impl.cpp:1889> [UID = 1]: deserialize engine from file :/home/eflag/Documents/yolov5/DeepStream-Yolo/model_b1_gpu0_fp32.engine failed
# 0:00:00.596714828 32181 0x563a8cd97120 WARN                 nvinfer gstnvinfer.cpp:635:gst_nvinfer_logger:<primary_gie> NvDsInferContext[UID 1]: Warning from NvDsInferContextImpl::generateBackendContext() <nvdsinfer_context_impl.cpp:1996> [UID = 1]: deserialize backend context from engine from file :/home/eflag/Documents/yolov5/DeepStream-Yolo/model_b1_gpu0_fp32.engine failed, try rebuild
# 0:00:00.596775144 32181 0x563a8cd97120 INFO                 nvinfer gstnvinfer.cpp:638:gst_nvinfer_logger:<primary_gie> NvDsInferContext[UID 1]: Info from NvDsInferContextImpl::buildModel() <nvdsinfer_context_impl.cpp:1914> [UID = 1]: Trying to create engine from model files

# Loading pre-trained weights
# Loading weights of /home/eflag/documents/yolov5/deepstream-yolo/v7/yolov7 complete
# Total weights read: 37669853
# Building YOLO network

#       layer                        input               output         weightPtr
# (0)   conv_silu                  3 x 640 x 640      32 x 640 x 640    992    
# (1)   conv_silu                 32 x 640 x 640      64 x 320 x 320    19680  
# (2)   conv_silu                 64 x 320 x 320      64 x 320 x 320    56800  
# (3)   conv_silu                 64 x 320 x 320     128 x 160 x 160    131040 
# (4)   conv_silu                128 x 160 x 160      64 x 160 x 160    139488 
# (5)   conv_silu                 64 x 160 x 160      64 x 160 x 160    143840 
# (6)   conv_silu                 64 x 160 x 160      64 x 160 x 160    180960 
# (7)   conv_silu                 64 x 160 x 160      64 x 160 x 160    218080 
# (8)   conv_silu                 64 x 160 x 160      64 x 160 x 160    255200 
# (9)   conv_silu                 64 x 160 x 160      64 x 160 x 160    292320 
# (10)  route                           -            128 x 160 x 160    292320 
# (11)  conv_silu                128 x 160 x 160     256 x 160 x 160    326112 
# (12)  conv_silu                256 x 160 x 160     128 x 160 x 160    359392 
# (13)  conv_silu                128 x 160 x 160     128 x 160 x 160    376288 
# (14)  conv_silu                128 x 160 x 160     128 x  80 x  80    524256 
# ERROR: [TRT]: 4: [layers.cpp::estimateOutputDims::1944] Error Code 4: Internal Error (route_15: all concat input tensors must have the same dimensions except on the concatenation axis (0), but dimensions mismatched at index 1. Input 0 shape: [128,80,80], Input 1 shape: [128,160,160])
# deepstream-app: utils.cpp:147: int getNumChannels(nvinfer1::ITensor*): Assertion `d.nbDims == 3' failed.
# ^C^C^C^CAborted (core dumped)




# eflag@eflag-MS-7D43:~/Documents/yolov5/DeepStream-Yolo$ deepstream-app -c deepstream_app_config.txt 
# gstnvtracker: Loading low-level lib at /opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so
# gstnvtracker: Batch processing is ON
# gstnvtracker: Past frame output is OFF
# [NvTrackerParams::getConfigRoot()] !!![WARNING] Empty config file path is provided. Will go ahead with default values
# [NvMultiObjectTracker] Initialized
# ERROR: ../nvdsinfer/nvdsinfer_model_builder.cpp:1484 Deserialize engine failed because file path: /home/eflag/Documents/yolov5/DeepStream-Yolo/model_b1_gpu0_fp16.engine open error
# 0:00:00.710273640  2805 0x55bd5365ad20 WARN                 nvinfer gstnvinfer.cpp:635:gst_nvinfer_logger:<primary_gie> NvDsInferContext[UID 1]: Warning from NvDsInferContextImpl::deserializeEngineAndBackend() <nvdsinfer_context_impl.cpp:1889> [UID = 1]: deserialize engine from file :/home/eflag/Documents/yolov5/DeepStream-Yolo/model_b1_gpu0_fp16.engine failed
# 0:00:00.721311397  2805 0x55bd5365ad20 WARN                 nvinfer gstnvinfer.cpp:635:gst_nvinfer_logger:<primary_gie> NvDsInferContext[UID 1]: Warning from NvDsInferContextImpl::generateBackendContext() <nvdsinfer_context_impl.cpp:1996> [UID = 1]: deserialize backend context from engine from file :/home/eflag/Documents/yolov5/DeepStream-Yolo/model_b1_gpu0_fp16.engine failed, try rebuild
# 0:00:00.721329204  2805 0x55bd5365ad20 INFO                 nvinfer gstnvinfer.cpp:638:gst_nvinfer_logger:<primary_gie> NvDsInferContext[UID 1]: Info from NvDsInferContextImpl::buildModel() <nvdsinfer_context_impl.cpp:1914> [UID = 1]: Trying to create engine from model files

# Loading pre-trained weights
# Loading weights of /home/eflag/documents/yolov5/deepstream-yolo/cam_18_22/best complete
# Total weights read: 7065607
# Building YOLO network

#       layer                        input               output         weightPtr
# (0)   conv_silu                  3 x 640 x 640      32 x 320 x 320    3584   
# (1)   conv_silu                 32 x 320 x 320      64 x 160 x 160    22272  
# (2)   conv_silu                 64 x 160 x 160      32 x 160 x 160    24448  
# (3)   route                           -             64 x 160 x 160    24448  
# (4)   conv_silu                 64 x 160 x 160      32 x 160 x 160    26624  
# (5)   conv_silu                 32 x 160 x 160      32 x 160 x 160    27776  
# (6)   conv_silu                 32 x 160 x 160      32 x 160 x 160    37120  
# (7)   shortcut_linear: 4              -             32 x 160 x 160        -  
# (8)   route                           -             64 x 160 x 160    37120  
# (9)   conv_silu                 64 x 160 x 160      64 x 160 x 160    41472  
# (10)  conv_silu                 64 x 160 x 160     128 x  80 x  80    115712 
# (11)  conv_silu                128 x  80 x  80      64 x  80 x  80    124160 
# (12)  route                           -            128 x  80 x  80    124160 
# (13)  conv_silu                128 x  80 x  80      64 x  80 x  80    132608 
# (14)  conv_silu                 64 x  80 x  80      64 x  80 x  80    136960 
# (15)  conv_silu                 64 x  80 x  80      64 x  80 x  80    174080 
# (16)  shortcut_linear: 13             -             64 x  80 x  80        -  
# (17)  conv_silu                 64 x  80 x  80      64 x  80 x  80    178432 
# (18)  conv_silu                 64 x  80 x  80      64 x  80 x  80    215552 
# (19)  shortcut_linear: 16             -             64 x  80 x  80        -  
# (20)  route                           -            128 x  80 x  80    215552 
# (21)  conv_silu                128 x  80 x  80     128 x  80 x  80    232448 
# (22)  conv_silu                128 x  80 x  80     256 x  40 x  40    528384 
# (23)  conv_silu                256 x  40 x  40     128 x  40 x  40    561664 
# (24)  route                           -            256 x  40 x  40    561664 
# (25)  conv_silu                256 x  40 x  40     128 x  40 x  40    594944 
# (26)  conv_silu                128 x  40 x  40     128 x  40 x  40    611840 
# (27)  conv_silu                128 x  40 x  40     128 x  40 x  40    759808 
# (28)  shortcut_linear: 25             -            128 x  40 x  40        -  
# (29)  conv_silu                128 x  40 x  40     128 x  40 x  40    776704 
# (30)  conv_silu                128 x  40 x  40     128 x  40 x  40    924672 
# (31)  shortcut_linear: 28             -            128 x  40 x  40        -  
# (32)  conv_silu                128 x  40 x  40     128 x  40 x  40    941568 
# (33)  conv_silu                128 x  40 x  40     128 x  40 x  40    1089536
# (34)  shortcut_linear: 31             -            128 x  40 x  40        -  
# (35)  route                           -            256 x  40 x  40    1089536
# (36)  conv_silu                256 x  40 x  40     256 x  40 x  40    1156096
# (37)  conv_silu                256 x  40 x  40     512 x  20 x  20    2337792
# (38)  conv_silu                512 x  20 x  20     256 x  20 x  20    2469888
# (39)  route                           -            512 x  20 x  20    2469888
# (40)  conv_silu                512 x  20 x  20     256 x  20 x  20    2601984
# (41)  conv_silu                256 x  20 x  20     256 x  20 x  20    2668544
# (42)  conv_silu                256 x  20 x  20     256 x  20 x  20    3259392
# (43)  shortcut_linear: 40             -            256 x  20 x  20        -  
# (44)  route                           -            512 x  20 x  20    3259392
# (45)  conv_silu                512 x  20 x  20     512 x  20 x  20    3523584
# (46)  conv_silu                512 x  20 x  20     256 x  20 x  20    3655680
# (47)  maxpool                  256 x  20 x  20     256 x  20 x  20    3655680
# (48)  maxpool                  256 x  20 x  20     256 x  20 x  20    3655680
# (49)  maxpool                  256 x  20 x  20     256 x  20 x  20    3655680
# (50)  route                           -           1024 x  20 x  20    3655680
# (51)  conv_silu               1024 x  20 x  20     512 x  20 x  20    4182016
# (52)  conv_silu                512 x  20 x  20     256 x  20 x  20    4314112
# (53)  upsample                 256 x  20 x  20     256 x  40 x  40        -  
# (54)  route                           -            512 x  40 x  40    4314112
# (55)  conv_silu                512 x  40 x  40     128 x  40 x  40    4380160
# (56)  route                           -            512 x  40 x  40    4380160
# (57)  conv_silu                512 x  40 x  40     128 x  40 x  40    4446208
# (58)  conv_silu                128 x  40 x  40     128 x  40 x  40    4463104
# (59)  conv_silu                128 x  40 x  40     128 x  40 x  40    4611072
# (60)  route                           -            256 x  40 x  40    4611072
# (61)  conv_silu                256 x  40 x  40     256 x  40 x  40    4677632
# (62)  conv_silu                256 x  40 x  40     128 x  40 x  40    4710912
# (63)  upsample                 128 x  40 x  40     128 x  80 x  80        -  
# (64)  route                           -            256 x  80 x  80    4710912
# (65)  conv_silu                256 x  80 x  80      64 x  80 x  80    4727552
# (66)  route                           -            256 x  80 x  80    4727552
# (67)  conv_silu                256 x  80 x  80      64 x  80 x  80    4744192
# (68)  conv_silu                 64 x  80 x  80      64 x  80 x  80    4748544
# (69)  conv_silu                 64 x  80 x  80      64 x  80 x  80    4785664
# (70)  route                           -            128 x  80 x  80    4785664
# (71)  conv_silu                128 x  80 x  80     128 x  80 x  80    4802560
# (72)  conv_silu                128 x  80 x  80     128 x  40 x  40    4950528
# (73)  route                           -            256 x  40 x  40    4950528
# (74)  conv_silu                256 x  40 x  40     128 x  40 x  40    4983808
# (75)  route                           -            256 x  40 x  40    4983808
# (76)  conv_silu                256 x  40 x  40     128 x  40 x  40    5017088
# (77)  conv_silu                128 x  40 x  40     128 x  40 x  40    5033984
# (78)  conv_silu                128 x  40 x  40     128 x  40 x  40    5181952
# (79)  route                           -            256 x  40 x  40    5181952
# (80)  conv_silu                256 x  40 x  40     256 x  40 x  40    5248512
# (81)  conv_silu                256 x  40 x  40     256 x  20 x  20    5839360
# (82)  route                           -            512 x  20 x  20    5839360
# (83)  conv_silu                512 x  20 x  20     256 x  20 x  20    5971456
# (84)  route                           -            512 x  20 x  20    5971456
# (85)  conv_silu                512 x  20 x  20     256 x  20 x  20    6103552
# (86)  conv_silu                256 x  20 x  20     256 x  20 x  20    6170112
# (87)  conv_silu                256 x  20 x  20     256 x  20 x  20    6760960
# (88)  route                           -            512 x  20 x  20    6760960
# (89)  conv_silu                512 x  20 x  20     512 x  20 x  20    7025152
# (90)  route                           -            128 x  80 x  80    7025152
# (91)  conv_logistic            128 x  80 x  80      45 x  80 x  80    7030957
# (92)  yolo                      45 x  80 x  80      45 x  80 x  80    7030957
# (93)  route                           -            256 x  40 x  40    7030957
# (94)  conv_logistic            256 x  40 x  40      45 x  40 x  40    7042522
# (95)  yolo                      45 x  40 x  40      45 x  40 x  40    7042522
# (96)  route                           -            512 x  20 x  20    7042522
# (97)  conv_logistic            512 x  20 x  20      45 x  20 x  20    7065607
# (98)  yolo                      45 x  20 x  20      45 x  20 x  20    7065607
# Output YOLO blob names: 
# yolo_93
# yolo_96
# yolo_99
# Total number of YOLO layers: 273
# Building YOLO network complete
# Building the TensorRT Engine

# NOTE: beta_nms is set in cfg file, make sure to set nms-iou-threshold=0.6 in config_infer file to get better accuracy

# WARNING: [TRT]: TensorRT was linked against cuBLAS/cuBLAS LT 11.6.3 but loaded cuBLAS/cuBLAS LT 11.5.4
# WARNING: [TRT]: TensorRT was linked against cuBLAS/cuBLAS LT 11.6.3 but loaded cuBLAS/cuBLAS LT 11.5.4
# Building complete

# 0:00:51.463292101  2805 0x55bd5365ad20 INFO                 nvinfer gstnvinfer.cpp:638:gst_nvinfer_logger:<primary_gie> NvDsInferContext[UID 1]: Info from NvDsInferContextImpl::buildModel() <nvdsinfer_context_impl.cpp:1947> [UID = 1]: serialize cuda engine to file: /home/eflag/Documents/yolov5/DeepStream-Yolo/model_b1_gpu0_fp32.engine successfully
# WARNING: [TRT]: TensorRT was linked against cuBLAS/cuBLAS LT 11.6.3 but loaded cuBLAS/cuBLAS LT 11.5.4
# INFO: ../nvdsinfer/nvdsinfer_model_builder.cpp:610 [Implicit Engine Info]: layers num: 4
# 0   INPUT  kFLOAT data            3x640x640       
# 1   OUTPUT kFLOAT yolo_93         45x80x80        
# 2   OUTPUT kFLOAT yolo_96         45x40x40        
# 3   OUTPUT kFLOAT yolo_99         45x20x20        

# 0:00:51.476408569  2805 0x55bd5365ad20 INFO                 nvinfer gstnvinfer_impl.cpp:313:notifyLoadModelStatus:<primary_gie> [UID 1]: Load new model:/home/eflag/Documents/yolov5/DeepStream-Yolo/config_infer_primary_yoloV5_custom.txt sucessfully

# Runtime commands:
#     h: Print this help
#     q: Quit

#     p: Pause
#     r: Resume

# NOTE: To expand a source in the 2D tiled display and view object details, left-click on the source.
#       To go back to the tiled display, right-click anywhere on the window.


# **PERF:  FPS 0 (Avg)    
# **PERF:  0.00 (0.00)    
# ** INFO: <bus_callback:194>: Pipeline ready

# ** INFO: <bus_callback:180>: Pipeline running

# **PERF:  277.51 (277.29)    
# **PERF:  263.48 (270.34)    
# **PERF:  267.00 (269.21)    
# **PERF:  263.05 (267.64)    
# **PERF:  261.44 (266.42)    
# **PERF:  265.62 (266.28)    
# ** INFO: <bus_callback:217>: Received EOS. Exiting ...

# Quitting
# [NvMultiObjectTracker] De-initialized
# App run successful