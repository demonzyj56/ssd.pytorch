import numpy as np
from vid.dataset import *


def load_gt_roidb(dataset_name, image_set_name, root_path, dataset_path, result_path=None,
                  flip=False):
    """ load ground truth roidb """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path, result_path)
    roidb = imdb.gt_roidb()
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb


def merge_roidb(roidbs):
    """ roidb are list, concat them together """
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    return roidb
