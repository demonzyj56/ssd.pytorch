""" ImagenNetVID loader for pytorch. """
import os
from bisect import bisect_left
import numpy as np
import cv2
import torch.utils.data
from vid.dataset.imagenet_vid import ImageNetVID, VID_CLASSES
from vid.utils.load_data import load_gt_roidb, merge_roidb, filter_roidb


_DEBUG_VISUALIZE = False


class VIDKeyframeDetection(torch.utils.data.Dataset):
    """ ImageNet VID Video Detection object.
    Reads keyframe and its annotations from image_sets,
    while loading its adjacent frames (but its annotations are not used).
    """

    def __init__(self, image_sets, root_path, dataset_path, result_path=None,
                 transform=None, dataset_name='ImageNetVID', is_test=False, k=0,
                 vid_context_set='VID_train_frames'):
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
            k: the size of context.
            vid_context_set: image set to find contexts, only appears at training.
        """
        assert transform is not None
        self.transform = transform
        self.name = dataset_name
        self.is_test = is_test
        self.k = k
        self.vid_context_set = vid_context_set
        if is_test:
            assert len(image_sets) == 1, "Should test only one dataset"
            self.imdb = ImageNetVID(image_sets[0], root_path, dataset_path, result_path)
            self.gt_roidb = self.imdb.gt_roidb()
            self.vid_context_roidb = None
        else:
            roidbs = [load_gt_roidb(dataset_name, iset, root_path, dataset_path,
                                    result_path, flip=False) for iset in image_sets]
            filtered_roidb = [filter_roidb(roidb) for roidb in roidbs]
            # No need to filter context roidb.
            print('Loading context roidb from {:s}...'.format(vid_context_set))
            self.vid_context_roidb = load_gt_roidb(dataset_name, vid_context_set,
                                                   root_path, dataset_path,
                                                   result_path, flip=False)
            # Not necessary...
            # self.vid_context_roidb.sort(key=lambda entry: entry['image'])
            filtered_roidb = [self._enrich_roidb_with_context(roidb, self.vid_context_roidb)
                              for roidb in filtered_roidb]
            self.gt_roidb = merge_roidb(filtered_roidb)

    def __len__(self):
        return len(self.gt_roidb)

    def __getitem__(self, index):
        """ Returns a pair of images and annotations for the first image.
        Return:
            image, image_pair, target
        """
        return self.pull_item_with_random_context(index)

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
        image_set = roi_rec['image'].split('/')[-2]
        return torch.from_numpy(img).permute(2, 0, 1), target, roi_rec['height'], roi_rec['width'], image_set

    def pull_item_with_random_context(self, index):
        """ Pulls an image together with its context frames.  The target is
        from the original image only. """
        roi_rec = self.gt_roidb[index]
        img = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR)
        select = np.random.randint(len(roi_rec['context']))
        img_ctx = cv2.imread(roi_rec['context'][select]['image'], cv2.IMREAD_COLOR)
        # Copy as a workaround for video annotation
        boxes = [roi_rec['boxes'], roi_rec['boxes'].copy()]
        labels = [roi_rec['gt_classes']-1, roi_rec['gt_classes'].copy()-1]
        # img and img_ctx own different memory
        imgs = [img, img_ctx]
        imgs, boxes, labels = self.transform(imgs, boxes, labels)
        # -------------- visualization ------------------
        # Note that the annotations are spread to both frames.
        if _DEBUG_VISUALIZE:
            import matplotlib.pyplot as plt
            mean = (104, 117, 123)
            for img, box, label in zip(imgs, boxes, labels):
                img_display = img.copy() + mean
                img_display[img_display > 255] = 255.
                img_display[img_display < 0] = 0.
                h, w, _ = img_display.shape
                plt.imshow(img_display[:, :, (2, 1, 0)] / 255)
                for j in range(box.shape[0]):
                    coords = (box[j, 0] * w, box[j, 1] * h), \
                             (box[j, 2] - box[j, 0]) * w, (box[j, 3] - box[j, 1]) * h
                    plt.gca().add_patch(plt.Rectangle(*coords, fill=False, linewidth=2, edgecolor='red'))
                    plt.gca().text(coords[0][0], coords[0][1], VID_CLASSES[label[j]],
                                   bbox={'facecolor': 'red', 'alpha': 0.5})
                plt.show()
        imgs = [torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1) for img in imgs]
        target = np.hstack((boxes[0], np.expand_dims(labels[0], axis=1)))

        return imgs[0], imgs[1], target

    def evaluate_detections(self, all_boxes):
        """ Evaluation code. """
        assert self.is_test
        return self.imdb.evaluate_detections(all_boxes)

    def _enrich_roidb_with_context(self, roidb, context):
        """ roidb is a dict holding various image paths and annotations.
        Enrich each entry with image paths from adjacent frames."""
        context_keys = [c['image'] for c in context]

        def find_pos_fast(entry):
            """ A (should-be) faster version using bisect.
            Refer to:
                https://code.activestate.com/recipes/577197-sortedcollection/
            """
            pos = bisect_left(context_keys, entry['image'])
            if pos != len(context_keys) and context_keys[pos] == entry['image']:
                return pos
            raise ValueError('The context set does not contain image: {}'.format(entry['image']))

        def adjacent(entry1, entry2):
            """ True if two entries are adjacent (belong to the same video sequence) but not the same.
            Pattern gives video folder name which should be the same. """
            return entry1['image'] != entry2['image'] and entry1['pattern'] == entry2['pattern']

        # we assume that context is sorted.
        # context field contains 2k entries (not containing itself) that should be adjacent to
        # the current entry.
        for entry in roidb:
            entry['context'] = []
            # Do so only if it is from VID.
            if not entry['det']:
                position = find_pos_fast(entry)
                beg = max(position-self.k, 0)
                end = min(position+self.k+1, len(context))
                for idx in range(beg, end):
                    if adjacent(entry, context[idx]):
                        entry['context'].append(context[idx])
            # If no context frames appear (i.e. DET images), pad itself.
            if len(entry['context']) == 0:
                entry['context'].append(entry)

        return roidb


def detection_collate_context(batch):
    """ The inputs are collated as the following:
    (img1, img_ctx1, img2, img_ctx2, ...), corresponding_targets.
    This is to fit for multigpu scenario."""
    images = []
    targets = []
    for img, img_ctx, target in batch:
        images.append(img)
        images.append(img_ctx)
        targets.append(torch.FloatTensor(target))
    return torch.stack(images, 0), targets
