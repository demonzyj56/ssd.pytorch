from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .vid15 import VIDDetection, VID_CLASSES
from .vid15_video import VIDVideoDetection
from .config import *
import cv2
import numpy as np


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    # x = cv2.resize(np.array(image), (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels

def base_transform_video(images, size, mean):
    imgs = []
    for img in images:
        x = cv2.resize(img, (size, size)).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        imgs.append(x)
    return imgs


class BaseTransform_video:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, images, boxes=None, labels=None):
        return base_transform_video(images, self.size, self.mean), boxes, labels