import numpy as np
import torch
import sys
import gdown
import os.path as osp

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker
from .deep.reid_model_factory import show_downloadeable_models, get_model_url, get_model_name

from torchreid.utils import FeatureExtractor
from torchreid.utils.tools import download_url

__all__ = ['StrongSORT']

# Pre-trained model from https://github.com/KaiyangZhou/deep-person-reid
DEFAULT_EXTRACTOR = ('osnet_x1_0', 'https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY')
EXTRACTOR_PATH = '%s_imagenet.pth' %(
    osp.abspath(osp.join(osp.dirname(__file__), "..", "weights", DEFAULT_EXTRACTOR[0])))


class StrongSORT(object):
    def __init__(
            self, 
            device, 
            max_dist = 0.2, 
            nn_budget = 100, 
            max_iou_distance = 0.7, 
            max_age = 70, 
            n_init = 3, 
            ema_alpha = 0.9, 
            mc_lambda = 0.995, 
            matching_cascade = False
        ):
        if not osp.isfile(EXTRACTOR_PATH):
            print(f'Feature extractor not found in {EXTRACTOR_PATH}')
            print(f'Downloading from torchreid model zoo...')
            print('    - Downloading {} from {}'.format(*DEFAULT_EXTRACTOR))
            print('    - To {}'.format(EXTRACTOR_PATH))
            download_url(DEFAULT_EXTRACTOR[1], EXTRACTOR_PATH)

        assert osp.isfile(EXTRACTOR_PATH), 'Couldn\'t load feature extractor.'

        self.extractor = FeatureExtractor(
            model_name=DEFAULT_EXTRACTOR[0], 
            model_path=EXTRACTOR_PATH, 
            device=str(device), 
            image_size=(256, 128), 
            pixel_norm=False, 
            verbose=False
        )

        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric("cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance, max_age, n_init, ema_alpha, mc_lambda, matching_cascade)

    def update(self, bbox_xywh, confidences, classes, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [
            Detection(classes[i], bbox_tlwh[i], conf, features[i]) \
            for i, conf in enumerate(confidences)
        ]

        # run on non-maximum supression
        # boxes = np.array([d.tlwh for d in detections]) ### ????
        # scores = np.array([d.confidence for d in detections]) ### ????

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes, confidences)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            
            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf], dtype='object'))
        try:
            return np.stack(outputs, axis=0)
        except ValueError:
            return np.zeros((0, 7), dtype='object')

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
