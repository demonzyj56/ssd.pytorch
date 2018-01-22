""" ImagenNetVID loader for pytorch. """
import os
import numpy as np
import cv2
import torch.utils.data
from vid.dataset.imagenet_vid import ImageNetVID, VID_CLASSES
from vid.utils.load_data import load_gt_roidb, merge_roidb, filter_roidb


class VIDPairDetection(torch.utils.data.Dataset):
    """ ImageNet VID Video Detection object.
    """

    def __init__(self, image_sets, root_path, dataset_path, result_path=None,
                 transform=None, dataset_name='ImageNetVID', is_test=False, k=0):
        """ Args:
            image_sets: a list of image set to use, which should correspond to txt files in
            data/.
            root_path: root path store cache and proposal data
            dataset_path: dataset path store images and image lists
            result_path: result path store results files.  If None, then set to
            root_path/cache/.
            transform: augmentations applied to image and label.  For testing,
            only BaseTransform is used.
            is_test: whether it is a test dataset.
            --k: the size of context
        """
        self.transform = transform
        self.name = dataset_name
        self.is_test = is_test
        self.k = k
        self.is_test = is_test
        if is_test:
            assert len(image_sets) == 1, "Should test only one dataset"
            self.imdb = ImageNetVID(image_sets[0], root_path, dataset_path, result_path)
            self.gt_roidb = self.imdb.gt_roidb()
        else:
            roidbs = [load_gt_roidb(dataset_name, iset, root_path, dataset_path,
                                    result_path, flip=False) for iset in image_sets]
            filtered_roidb = [filter_roidb(roidb) for roidb in roidbs]
            self.gt_roidb = merge_roidb(filtered_roidb)    # gt_roidb records all information from the imageset.txt file
        self.tot_len = len(self.gt_roidb)

    def __len__(self):
        return len(self.gt_roidb)

    def __getitem__(self, index):
        """ Returns a pair of images and their corresponding annotations.
        Return:
            (image1, target1), (image2, target2).
        """
        imgs = []  # now we extract a context of current frame for batch transformation/augmentation
        im, gt, _, _ = self.pull_orig_item(index=index, is_context=False)
        # plt.imshow(im)
        # plt.show()

        imgs.append(im)
        # determine the offset from current frame
        repeat_times = 5
        for i in range(repeat_times):
            offset = np.random.randint(1, 1 + self.k)
            if np.random.randint(2):  # randomly pick direction
                offset = -offset
            if self.exist(index, offset):
                im, gt_pair, _, _ = self.pull_orig_item(index + offset, is_context=False)
                imgs.append(im)
                break
            elif i==repeat_times-1:
                imgs.append(imgs[-1])
                gt_pair = gt

        # -------------- visualization ------------------
        # from matplotlib import pyplot as plt
        # plt.imshow(imgs[0][:, :, (2, 1, 0)]); plt.show()
        # plt.imshow(imgs[1][:, :, (2, 1, 0)]); plt.show()

        if self.transform != None:
            image1, boxes1, labels1 = self.transform(imgs[0], gt[0], gt[1]-1)
            image2, boxes2, labels2 = self.transform(imgs[1], gt_pair[0], gt_pair[1]-1)
            image1 = torch.from_numpy(image1[:, :, (2, 1, 0)]).permute(2, 0, 1)
            image2 = torch.from_numpy(image2[:, :, (2, 1, 0)]).permute(2, 0, 1)
            target1 = np.hstack((boxes1, np.expand_dims(labels1, axis=1)))
            target2 = np.hstack((boxes2, np.expand_dims(labels2, axis=1)))

        return (image1, target1), (image2, target2)

        # num = gt[0].shape[0]
        # gt[0] = np.concatenate((gt[0], gt_pair[0]), axis=0)
        # gt[1] = np.concatenate((gt[1], gt_pair[1]), axis=0)
        #
        # if self.transform != None:
        #     imgs, boxes, labels = self.transform(imgs, gt[0], gt[1]-1)
        #     im = []
        #     for i in range(2):
        #         im.append(torch.from_numpy(imgs[i][:, :, (2, 1, 0)]).permute(2, 0, 1))
        #     target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        #     target = (target[:num, :], target[num:, :])
        #
        # imgs = torch.stack(im, 0)
        #
        # return imgs, target


    def pull_item(self, index):
        """ Mimic VOCDetection.
            Note that roi['gt_classes'] is np.int32 but seem to have no effect.
            Note that roi['gt_classes'] ranges in {1, ..., 30} and background is index 0 (but never used).
            To fit for current version of SSD where gt should in range(0, 30),
            the label will be simply shifted by 1.
        """
        roi_rec = self.gt_roidb[index]
        img = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR)
        if self.transform is not None:
            img, boxes, labels = self.transform(img, roi_rec['boxes'], roi_rec['gt_classes']-1)
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, roi_rec['height'], roi_rec['width']

    def pull_orig_item(self, index, is_context):
        """
           pull images and targets without transformation, for later video transformation
        """
        roi_rec = self.gt_roidb[index]
        assert os.path.exists(roi_rec['image']), "Non-existing path: {}".format(roi_rec['image'])
        img = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR)
        if is_context == False:
            target = [roi_rec['boxes'], roi_rec['gt_classes']] # a simple list to warp them for later processing
            return img, target, roi_rec['height'], roi_rec['width']
        else:
            return img, roi_rec['height'], roi_rec['width']

    def exist(self, index, offset):
        '''
        Determine whether a context frame exist in the imageset txt file
        :param index: absolute index of current frame
        :param offset: frame index offset to current frame
        :return: whether the context frame exists
        '''
        # absolute index range
        if self.is_test == False:
            if 'pattern' in self.gt_roidb[index].keys():
                max_idx = self.tot_len-1
                min_idx = 0

                context_idx = index + offset
                    # if exist, pattern must be the same
                if context_idx<min_idx or context_idx>max_idx:
                    return False
                else:
                    if 'pattern' in self.gt_roidb[context_idx].keys():
                        if self.gt_roidb[index]['pattern'] != self.gt_roidb[context_idx]['pattern']:
                            return False
                        elif self.gt_roidb[index]['frame_seg_id'] + offset != self.gt_roidb[context_idx]['frame_seg_id']:
                            return False
                        else:
                            return True
                    else:
                        return False
            else:
                return False
        else:
            max_idx = self.tot_len - 1
            min_idx = 0
            context_idx = index+offset
            if context_idx<min_idx or context_idx>max_idx:
                return False
            else:
                pattern = self.gt_roidb[index]['image'].split('/')[-2]
                context_pattern = self.gt_roidb[context_idx]['image'].split('/')[-2]
                if pattern != context_pattern:
                    return False
                else:
                    return True


    def evaluate_detections(self, all_boxes):
        """ Evaluation code. """
        assert self.is_test
        return self.imdb.evaluate_detections(all_boxes)


# def detection_collate_pair(batch):
#     """Custom collate fn for dealing with batches of images that have a different
#     number of associated object annotations (bounding boxes).
#
#     Arguments:
#         batch: (tuple) A tuple of tensor images and lists of annotations
#
#     Return:
#         A tuple containing:
#             1) (tensor) batch of images stacked on their 0 dim
#             2) (list of tensors) annotations for a given image are stacked on 0 dim
#     """
#     targets = []
#     imgs = []
#     for sample in batch:
#         imgs.append(sample[0])
#         targets.append(torch.FloatTensor(sample[1][0]))
#         targets.append(torch.FloatTensor(sample[1][1]))
#     return torch.stack(imgs, 0), targets

def detection_collate_pair(batch):
    """ The inputs are collated as the following:
    cat(image_tensor, image_pair_tensor), corresponding_targets. """
    image = []
    image_pair = []
    target = []
    target_pair = []
    for sample, sample_pair in batch:
        image.append(sample[0])
        image_pair.append(sample_pair[0])
        target.append(torch.FloatTensor(sample[1]))
        target_pair.append(torch.FloatTensor(sample_pair[1]))
    return torch.stack(image+image_pair, 0), target+target_pair

