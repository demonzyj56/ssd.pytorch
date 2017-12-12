""" ImagenNetVID loader for pytorch. """
import numpy as np
import cv2
import torch.utils.data
from vid.dataset.imagenet_vid import ImageNetVID, VID_CLASSES
from vid.utils.load_data import load_gt_roidb, merge_roidb, filter_roidb


class VIDDetection(torch.utils.data.Dataset):
    """ ImageNet VID Video Detection object. """

    def __init__(self, image_sets, root_path, dataset_path, result_path=None,
                 transform=None, dataset_name='ImageNetVID', is_test=False):
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


if __name__ == "__main__":
    from utils.augmentations import SSDAugmentation
    vid_det = VIDDetection('DET_train_30classes+VID_train_15frames', 'data/', '/home/leoyolo/research/data/ILSVRC',
                           transform=SSDAugmentation())
    from IPython import embed; embed()
