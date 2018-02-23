""" Analyse the positive distributions for give dataset. """
import argparse
from pprint import pprint
import numpy as np
import torch
import torch.utils.data as tdata
from layers.box_utils import match
from layers.functions import PriorBox
from data import v2_512, AnnotationTransform, VOCDetection, VOCroot, \
    VOC_CLASSES, VIDroot, VIDDetection, VID_CLASSES, detection_collate
from utils.augmentations import SSDAugmentation


def prepare_positive_targets(priors, targets, batch_size, ov_thresh, cfg):
    """ Given priors and targets for an image, return the ground truth
    locations which is ordered as [batch_size, num_priors], and each
    entry is 1 if it is a postive sample.
    """
    conf_t= torch.LongTensor(batch_size, priors.size(0)).cuda()
    loc_t = torch.Tensor(batch_size, priors.size(0), 4).cuda()
    for idx in range(batch_size):
        truths = targets[idx][:, :-1]
        labels = targets[idx][:, -1]
        match(ov_thresh, truths, priors, cfg['variance'], labels,
              loc_t, conf_t, idx)
    conf_t = conf_t.gt(0)

    return conf_t


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='voc',
                        help='VID or VOC')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='Overlap threshold')
    parser.add_argument('--augment', default=True, type=bool,
                        help='Whether the dataset is augmented')
    args = parser.parse_args()
    print('Args:')
    pprint(args)

    return args


if __name__ == '__main__':
    args = parse_args()
    ssd_dim = 512
    means = (104, 117, 123)
    priors = PriorBox(v2_512).forward().cuda()
    if args.augment:
        transform = SSDAugmentation(ssd_dim, means)
    else:
        transform = BaseTransform(ssd_dim, means)
    if args.dataset == 'vid':
        train_sets = ['DET_train_30classes', 'VID_train_15frames']
        dataset = VIDDetection(train_sets, 'data/', VIDroot,
                               transform=transform, is_test=False)
    else:
        train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
        dataset = VOCDetection(VOCroot, train_sets, transform,
                               AnnotationTransform())
    num_pos = np.zeros((len(dataset),), dtype=np.float32)
    for idx, (_, target) in enumerate(dataset):
        target = [torch.FloatTensor(target).cuda()]
        conf_t = prepare_positive_targets(priors, target, 1, args.threshold,
                                          v2_512)
        num_pos[idx] = conf_t.sum()
        print('Number of positive for {}/{}: {}'.format(
            idx+1, len(dataset), conf_t.sum()
        ))

    print('Number of positives on average for {}: {}'.format(args.dataset,
                                                             num_pos.mean()))

