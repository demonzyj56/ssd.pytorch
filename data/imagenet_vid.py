""" ImagenNetVID loader for pytorch. """
import cv2
import torch.utils.data
from vid.dataset import ImageNetVID
from vid.utils.load_data import load_gt_roidb, merge_roidb


class VIDDetection(torch.utils.data.Dataset):
    """ ImageNet VID Video Detection object. """

    def __init__(self, image_set, root_path, dataset_path, result_path=None,
                 transform=None, dataset_name='ImageNetVID'):
        image_sets = [iset for iset in image_set.split('+')]
        roidbs = [load_gt_roidb(dataset_name, iset, root_path, dataset_path,
                                result_path, flip=False) for iset in image_sets]
        self.gt_roidb = merge_roidb(roidbs)
        self.transform = transform

    def __len__(self):
        return len(self.gt_roidb)

    def __getitem__(self, index):
        img, target, _ = self.pull_item(index)
        return img, target

    def pull_item(self, index):
        """ Mimic VOCDetection.
        TODO(leoyolo): the gt_roidb given by IMDB has the following problem:
            - boxes coordinates are in absolute ones instead of fractional coordinates (i.e. not divided by height/width).
            - boxes and labels have different data types in numpy.
        """
        roi_rec = self.gt_roidb[index]
        img = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR)
        if self.transform is not None:
            img, boxes, labels = self.transform(img, roi_rec['boxes'], roi_rec['gt_classes'])
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, roi_rec['frame_id']

