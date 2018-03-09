""" Calculates overlapping positive locations in consecutive frames. """
import argparse
from pprint import pprint
import numpy as np
import torch
import torch.utils.data as tdata
from layers.box_utils import match
from layers.functions import PriorBox
from data import v2_512, AnnotationTransform, VIDroot, VID_CLASSES, BaseTransform
from data.vid15_keyframe import VIDKeyframeDetection, detection_collate_context
from utils.augmentations_video import SSDAugmentationVideo


mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
}


def prepare_positive_targets(priors, targets, batch_size, ov_thresh, cfg):
    """ Given priors and targets for an image, do the following:
    Find the location of each positive targets, then unmap into separate
    masks for each feature map.  Each feature map is of size
    [batch_size, mbox_size, height, width].
    """
    conf_t= torch.LongTensor(batch_size, priors.size(0)).cuda()
    loc_t = torch.Tensor(batch_size, priors.size(0), 4).cuda()
    for idx in range(batch_size):
        truths = targets[idx][:, :-1]
        labels = targets[idx][:, -1]
        defaults = priors
        match(ov_thresh, truths, defaults, cfg['variance'], labels,
              loc_t, conf_t, idx)

    feature_segs = zip(cfg['feature_maps'], mbox[str(cfg['min_dim'])])
    feature_size = [i ** 2 * j for (i, j) in feature_segs]
    assert sum(feature_size) == conf_t.size(1), \
        'Total feature size is {} while conf_t gives {}'.format(sum(feature_size), conf_t.size(1))
    gt_masks = []
    accum = 0
    for (i, j) in zip(cfg['feature_maps'], mbox[str(cfg['min_dim'])]):
        gt_masks.append(
                conf_t[:, accum:(accum+i**2*j)].contiguous().view(-1, j, i, i)
            )
        accum += i ** 2 * j
    return gt_masks


def matched_each_level(mask1, mask2):
    """ Returns the number of matched location for two masks on the
    same level. """
    return (mask1.gt(0) & mask2.gt(0) & mask1.eq(mask2)).sum()


def num_matched(gt_mask1, gt_mask2):
    """ Returns the number of matched locations for two masks for two
    consecutive frames. """
    return sum([
        matched_each_level(m1, m2) for m1, m2 in zip(gt_mask1, gt_mask2)
    ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='Overlap threshold')
    parser.add_argument('--augment', default=False, type=bool,
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
        transform = SSDAugmentationVideo(ssd_dim, means)
    else:
        transform = BaseTransform(ssd_dim, means)
    train_sets = ['VID_train_frames']
    dataset = VIDKeyframeDetection(train_sets, 'data/', VIDroot,
                                   transform=transform, is_test=False)
    matched = np.zeros((len(dataset), ), dtype=np.int32)
    _, target, _, _, cached_video_name = dataset.pull_item(0)  # the first one
    target = [torch.FloatTensor(target).cuda()]
    cached_mask = prepare_positive_targets(priors, target, 1, args.threshold, v2_512)
    for idx in range(1, len(dataset)):
        _, target, _, _, video_name = dataset.pull_item(idx)
        target = [torch.FloatTensor(target).cuda()]
        gt_mask = prepare_positive_targets(priors, target, 1, args.threshold, v2_512)
        if video_name == cached_video_name:
            n = num_matched(cached_mask, gt_mask)
            print('Number of matched samples for {:d}/{:d}: {:d}'.format(
                idx+1, len(dataset), n
            ))
            matched[idx] = n
        cached_mask = gt_mask
        cached_video_name = video_name

    from IPython import embed; embed()
