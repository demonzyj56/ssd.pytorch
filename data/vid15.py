""" ImagenNetVID loader for pytorch. """
import numpy as np
import cv2
import torch.utils.data
from vid.dataset.imagenet_vid import ImageNetVID, VID_CLASSES
from vid.utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from utils.augmentations import ToAbsoluteCoords, RandomSampleCrop, ToPercentCoords, Compose


class VIDDetection(torch.utils.data.Dataset):
    """ ImageNet VID Video Detection object. """

    def __init__(self, image_sets, root_path, dataset_path, result_path=None,
                 transform=None, dataset_name='ImageNetVID', is_test=False, custom_filter=None):
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
            custom_filter: if specified, then filter each roidb before merging (only for training).
        """
        self.transform = transform
        self.name = dataset_name
        self.is_test = is_test
        if is_test:
            assert len(image_sets) == 1, "Should test only one dataset"
            self.imdb = ImageNetVID(image_sets[0], root_path, dataset_path, result_path)
            self.gt_roidb = self.imdb.gt_roidb()
        else:
            roidbs = [load_gt_roidb(dataset_name, iset, root_path, dataset_path,
                                    result_path, flip=False) for iset in image_sets]
            filtered_roidb = [filter_roidb(roidb) for roidb in roidbs]
            if custom_filter is not None:
                filtered_roidb = [custom_filter(roidb) for roidb in filtered_roidb]
            self.gt_roidb = merge_roidb(filtered_roidb)


    def __len__(self):
        return len(self.gt_roidb)

    def __getitem__(self, index):
        img, target, _, _ = self.pull_item(index)
        return img, target

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

    def evaluate_detections(self, all_boxes):
        """ Evaluation code. """
        assert self.is_test
        return self.imdb.evaluate_detections(all_boxes)


class VIDVideoDetection(torch.utils.data.Dataset):
    """ Another version of vid object detection with context. """

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
            k: the size of context
        """
        self.transform = transform
        self.name = dataset_name
        self.is_test = is_test
        self.k = k
        if is_test:
            assert len(image_sets) == 1, "Should test only one dataset"
            self.imdb = ImageNetVID(image_sets[0], root_path, dataset_path, result_path)
            self.gt_roidb = self.imdb.gt_roidb()
        else:
            roidbs = [load_gt_roidb(dataset_name, iset, root_path, dataset_path,
                                    result_path, flip=False) for iset in image_sets]
            filtered_roidb = [filter_roidb(roidb) for roidb in roidbs]
            self.gt_roidb = merge_roidb(filtered_roidb)
        # TODO(leoyolo): This may create too large gap.
        self.det_random_crop = Compose([
            ToAbsoluteCoords(),
            RandomSampleCrop(),
            ToPercentCoords()
        ])

    def __len__(self):
        """ Number of frames (DET+VID). """
        return len(self.gt_roidb)

    def __getitem__(self, index):
        """ Use self.pull_item_video to dispatch to DET/VID. """
        imgs, target, _, _ = self.pull_item_video(index)

        return imgs, target

    def pull_item_video(self, index):
        """ Dispatch to DET/VID. """
        if self.check_is_det(index):
            return self.pull_item_det(index)
        else:
            return self.pull_item_vid(index)

    def pull_item_vid(self, index):
        """ Pull an item from VID dataset.  For VID branch, we implement the following:
        """
        roi_rec = self.gt_roidb[index]
        imgs = []
        imgs.append(cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR))
        flag_pre = True
        for offset in range(1, self.k+1):
            if flag_pre and self.exist(index, offset):
                img = cv2.imread(self.gt_roidb[index+offset]['image'], cv2.IMREAD_COLOR)
                imgs.append(img)
            else:
                imgs.append(imgs[-1])
                flag_pre = False
        flag_post = True
        for offset in range(-1, -self.k-1, -1):
            if flag_post and self.exist(index, offset):
                img = cv2.imread(self.gt_roidb[index+offset]['image'], cv2.IMREAD_COLOR)
                imgs.insert(0, img)
            else:
                imgs.insert(0, imgs[0])
                flag_post = False

        imgs, boxes, labels = self.transform(imgs, roi_rec['boxes'], roi_rec['gt_classes']-1)
        imgs = torch.stack(
            [torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1) for img in imgs],
            0
        )
        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return imgs, target, roi_rec['height'], roi_rec['width']

    def pull_item_det(self, index):
        """ Pull an item from DET dataset.  For DET branch, we manually
        create the motion by random cropping the static images and concatenate
        it as some pseudo-motion. """
        roi_rec = self.gt_roidb[index]
        boxes = roi_rec['boxes']
        labels = roi_rec['gt_classes'] - 1
        imgs = []
        img = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR)
        for _ in range(self.k):
            cropped_img, _, _ = self.det_random_crop(img, boxes, labels)
            imgs.append(cropped_img)
        imgs.append(img)
        for _ in range(self.k):
            cropped_img, _, _ = self.det_random_crop(img, boxes, labels)
            imgs.append(cropped_img)
        # Note that the images cropped in the previous step have different sizes
        # but transform will force them to be the same.
        imgs, boxes, labels = self.transform(imgs, boxes, labels)
        imgs = torch.stack(
            [torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1) for img in imgs],
            0
        )
        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return imgs, target, roi_rec['height'], roi_rec['width']

    def exist(self, index, offset):
        '''
        Determine whether a context frame exist in the imageset txt file
        :param index: absolute index of current frame
        :param offset: frame index offset to current frame
        :return: whether the context frame exists
        '''
        context_idx = index + offset
        if context_idx < 0 or context_idx >= len(self):
            return False
        # absolute index range
        if not self.is_test:
            # VID case, need to look carefully
            if not self.check_is_det(index):
                # if exist, pattern must be the same
                if self.gt_roidb[index]['pattern'] != self.gt_roidb[context_idx]['pattern']:
                    return False
                elif self.gt_roidb[index]['frame_seg_id'] + offset != self.gt_roidb[context_idx]['frame_seg_id']:
                    return False
                else:
                    return True
            else:
                # Just ignore
                return False
        else:
            pattern = self.gt_roidb[index]['image'].split('/')[-2]
            context_pattern = self.gt_roidb[context_idx]['image'].split('/')[-2]
            if pattern != context_pattern:
                return False
            else:
                return True

    def check_is_det(self, index):
        """ Check whether a specific roidb is from DET or VID. """
        return self.gt_roidb[index]['det']


if __name__ == "__main__":
    from utils.augmentations import SSDAugmentation
    vid_det = VIDDetection('DET_train_30classes+VID_train_15frames', 'data/', '/home/leoyolo/research/data/ILSVRC',
                           transform=SSDAugmentation())
    from IPython import embed; embed()
