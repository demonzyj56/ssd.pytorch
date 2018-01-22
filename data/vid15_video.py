""" ImagenNetVID loader for pytorch. """
import os
import numpy as np
import cv2
import torch.utils.data
from vid.dataset.imagenet_vid import ImageNetVID, VID_CLASSES
from vid.utils.load_data import load_gt_roidb, merge_roidb, filter_roidb


class VIDVideoDetection(torch.utils.data.Dataset):
    """ ImageNet VID Video Detection object.
        Image items are pulled with their context
        !!!Only applicable to VID dataset
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
        # img, target, _, _ = self.pull_item(index)
        # ---------------------------------------------
        imgs = []  # now we extract a context of current frame for batch transformation/augmentation
        im, gt, _, _ = self.pull_orig_item(index=index, is_context=False)
        imgs.append(im)
        flag = True # used to enforce consistency of fetching frames
        for offset in range(1, self.k+1):
            if self.exist(index, offset) and flag:
                im, _, _ = self.pull_orig_item(index + offset, is_context=True)
                imgs.append(im)
            else:
                imgs.append(imgs[-1])
                flag = False
        flag = True
        for offset in range(-1, -self.k-1, -1):
            if self.exist(index, offset) and flag:
                im, _, _ = self.pull_orig_item(index + offset, is_context=True)
                imgs.insert(0, im)
            else:
                imgs.insert(0, imgs[0])
                flag = False

        if self.transform != None: # only video transformation is applicable
            imgs, boxes, labels = self.transform(imgs, gt[0], gt[1]-1)
            if isinstance(imgs, list):
                # Returns image sequences
                im = []
                for i in range(self.k*2+1):
                    im.append(torch.from_numpy(imgs[i][:, :, (2, 1, 0)]).permute(2, 0, 1))
                    imgs = torch.stack(im, 0)
            else:
                # Returns only single frame
                imgs = torch.from_numpy(imgs[:, :, (2, 1, 0)]).permute(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return imgs, target


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

    def pull_item_video(self, index):
        imgs = []  # now we extract a context of current frame for batch transformation/augmentation
        im, gt, h, w = self.pull_orig_item(index=index, is_context=False)
        imgs.append(im)
        flag = True # used to enforce consistency of fetching frames
        for offset in range(1, self.k+1):
            if self.exist(index, offset) and flag:
                im, _, _ = self.pull_orig_item(index + offset, is_context=True)
                imgs.append(im)
            else:
                imgs.append(imgs[-1])
                flag = False
        flag = True
        for offset in range(-1, -self.k-1, -1):
            if self.exist(index, offset) and flag:
                im, _, _ = self.pull_orig_item(index + offset, is_context=True)
                imgs.insert(0, im)
            else:
                imgs.insert(0, imgs[0])
                flag = False

        if self.transform != None: # only video transformation is applicable
            imgs, boxes, labels = self.transform(imgs, gt[0], gt[1]-1)
            if isinstance(imgs, list):
                # Returns image sequences
                im = []
                for i in range(self.k*2+1):
                    im.append(torch.from_numpy(imgs[i][:, :, (2, 1, 0)]).permute(2, 0, 1))
                    imgs = torch.stack(im, 0)
            else:
                # Returns only single frame
                imgs = torch.from_numpy(imgs[:, :, (2, 1, 0)]).permute(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return imgs, target, h, w

    def pull_orig_item(self, index, is_context):
        """
           pull images and targets without transformation, for later video transformation
        """
        roi_rec = self.gt_roidb[index]
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
                    if not ('pattern' in self.gt_roidb[context_idx].keys()):
                        return False
                    if self.gt_roidb[index]['pattern'] != self.gt_roidb[context_idx]['pattern']:
                        return False
                    elif self.gt_roidb[index]['frame_seg_id'] + offset != self.gt_roidb[context_idx]['frame_seg_id']:
                        return False
                    else:
                        return True
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


if __name__ == "__main__":
    from utils.augmentations import SSDAugmentation
    vid_det = VIDVideoDetection('DET_train_30classes+VID_train_15frames', 'data/', '/home/leoyolo/research/data/ILSVRC',
                           transform=SSDAugmentation())
    from IPython import embed; embed()
