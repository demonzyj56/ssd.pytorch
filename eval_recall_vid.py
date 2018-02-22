import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VIDroot
from data import VID_CLASSES as labelmap
from data import VIDDetection, BaseTransform
from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--vid_val_list', default='VID_val_frames',
                    help='validation files found in ${vid_root}/${vid_val_list}.txt')
parser.add_argument('--size', default=512, type=int, help='Which size to use (300/512)')
parser.add_argument('--class_specific', default=True, type=str2bool, help='Choose class to evaluate')
args = parser.parse_args()

dataset_mean = (104, 117, 123)

detections = pickle.load(open('ssd512_120000/test/detections.pkl', 'rb'))
dataset = VIDDetection([args.vid_val_list], 'data/', VIDroot,
                       transform=BaseTransform(args.size, dataset_mean), is_test=True)
if args.class_specific:
    for class_name in labelmap:
        print('================ {} ==================='.format(class_name))
        class_to_index = dict(zip(labelmap, range(len(labelmap))))
        detections_class = detections[class_to_index[class_name] + 1]
        dataset.imdb.evaluate_recall_class_specific(detections_class, class_name, thresholds=[0.5])
else:
    dataset.imdb.evaluate_recall_from_detections(detections, thresholds=None)
