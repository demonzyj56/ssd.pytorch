import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

class ConvertFromInts_video(object):
    def __call__(self, images, boxes=None, labels=None):
        imgs = []
        for i in range(len(images)):
            imgs.append(images[i].astype(np.float32))
        return imgs, boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels

class SubtractMeans_video(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, images, boxes=None, labels=None):
        for i in range(len(images)):
            images[i] = images[i].astype(np.float32)
            images[i] -= self.mean
            images[i] = images[i].astype(np.float32)
        return images, boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels

class ToAbsoluteCoords_video(object):
    def __call__(self, images, boxes=None, labels=None):
        """ Do separately for each image and annotation pair. """
        for idx, img in enumerate(images):
            height, width, channels = img.shape
            boxes[idx][:, 0] *= width
            boxes[idx][:, 2] *= width
            boxes[idx][:, 1] *= height
            boxes[idx][:, 3] *= height

        return images, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels

class ToPercentCoords_video(object):
    def __call__(self, images, boxes=None, labels=None):
        """ Do separately for each image and annotation pair. """
        for idx, img in enumerate(images):
            height, width, channels = img.shape
            boxes[idx][:, 0] /= width
            boxes[idx][:, 2] /= width
            boxes[idx][:, 1] /= height
            boxes[idx][:, 3] /= height

        return images, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        if type(self.size)==type(300):
            image = cv2.resize(image, (self.size,
                                    self.size))
        else:
            image = cv2.resize(image, (self.size[0],
                                       self.size[1]))
        return image, boxes, labels


class Resize_video(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, images, boxes=None, labels=None):
        if type(self.size)==type(300):
            for i in range(len(images)):
                images[i] = cv2.resize(images[i], (self.size,
                                        self.size))
        else:
            for i in range(len(images)):
                images[i] = cv2.resize(images[i], (self.size[0],
                                               self.size[1]))
        return images, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels

class RandomSaturation_video(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, images, boxes=None, labels=None):
        if random.randint(2):
            rnd_num = random.uniform(self.lower, self.upper)
            for i in range(len(images)):
                images[i][:, :, 1] *= rnd_num

        return images, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels

class RandomHue_video(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, images, boxes=None, labels=None):
        if random.randint(2):
            rnd_num = random.uniform(-self.delta, self.delta)
            for i in range(len(images)):
                images[i][:, :, 0] += rnd_num
                images[i][:, :, 0][images[i][:, :, 0] > 360.0] -= 360.0
                images[i][:, :, 0][images[i][:, :, 0] < 0.0] += 360.0
        return images, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels

class RandomLightingNoise_video(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, images, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            for i in range(len(images)):
                images[i] = shuffle(images[i])
        return images, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels

class ConvertColor_video(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, images, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            for i in range(len(images)):
                images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            for i in range(len(images)):
                images[i] = cv2.cvtColor(images[i], cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return images, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels

class RandomContrast_video(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, images, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            for i in range(len(images)):
                images[i] *= alpha
        return images, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels

class RandomBrightness_video(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, images, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            for i in range(len(images)):
                images[i] += delta
        return images, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class RandomSampleCrop_video(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self, sample_options=None):
        """ Allows custom sample options to be used. """
        if sample_options is None:
            self.sample_options = (
                # using entire original input image
                None,
                # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
                (0.1, None),
                (0.3, None),
                (0.7, None),
                (0.9, None),
                # randomly sample a patch
                (None, None),
            )
        else:
            self.sample_options = sample_options

    def __call__(self, images, boxes=None, labels=None):
        """ We assumes that all image frames have equal size, which is
        reasonable for videos.
        For videos, we enforce a more strict cropping rule that the sample options
        should be wrt unions of all boxes.
        """
        height, width, _ = images[0].shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return images, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(np.concatenate(boxes, axis=0), rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou or max_iou < overlap.max():
                    continue

                # cut the crop from the image
                # current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],:]
                current_images = []
                for i in range(len(images)):
                    current_images.append(images[i][rect[1]:rect[3], rect[0]:rect[2],
                                              :])

                box_masks = []
                for box in boxes:
                    # keep overlap with gt box IF center in sampled patch
                    centers = (box[:, :2] + box[:, 2:]) / 2.0

                    # mask in all gt boxes that above and to the left of centers
                    m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                    # mask in all gt boxes that under and to the right of centers
                    m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                    # mask in that both m1 and m2 are true
                    mask = m1 * m2

                    box_masks.append(mask)

                # try again if all box is not valid
                if not all([m.any() for m in box_masks]):
                    continue

                current_boxes = []
                current_labels = []
                for box, mask, label in zip(boxes, box_masks, labels):
                    # take only matching gt boxes
                    current_box = box[mask, :].copy()

                    # take only matching gt labels
                    current_label = label[mask]

                    # should we use the box left and top corner or the crop's
                    current_box[:, :2] = np.maximum(current_box[:, :2], rect[:2])

                    # adjust to crop (by substracting crop's left,top)
                    current_box[:, :2] -= rect[:2]

                    current_box[:, 2:] = np.minimum(current_box[:, 2:], rect[2:])

                    # adjust to crop (by substracting crop's left,top)
                    current_box[:, 2:] -= rect[:2]

                    current_boxes.append(current_box)
                    current_labels.append(current_label)

                return current_images, current_boxes, current_labels



class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels

class Expand_video(object):
    def __init__(self, mean, expand_factor=None):
        self.mean = mean
        if expand_factor is None:
            self.expand_factor = 4
        else:
            self.expand_factor = expand_factor

    def __call__(self, images, boxes, labels):
        """ We actually assumes that all images have the same shape, which is
        fair for videos.  Expand the frames and boxes by the same ratio.
        Also notice that Expand assumes absolute coordinate. """
        if random.randint(2):
            return images, boxes, labels

        height, width, depth = images[0].shape
        ratio = random.uniform(1, self.expand_factor)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=images[0].dtype)
        expand_image[:, :, :] = self.mean

        imgs = []
        expand_boxes = []
        for i in range(len(images)):
            tmp_img = expand_image.copy()
            tmp_img[int(top):int(top + height),
                         int(left):int(left + width)] = images[i]
            imgs.append(tmp_img)
            box = boxes[i].copy()
            box[:, :2] += (int(left), int(top))
            box[:, 2:] += (int(left), int(top))
            expand_boxes.append(box)
        images = imgs

        return images, expand_boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes

class RandomMirror_video(object):
    def __call__(self, images, boxes, classes):
        """ Notice that random mirror assumes absolute coordinate. """
        if random.randint(2):
            mirrored_boxes = []
            for i in range(len(images)):
                width = images[i].shape[1]
                images[i] = images[i][:, ::-1]
                box = boxes[i].copy()
                box[:, 0::2] = width - box[:, 2::-2]
                mirrored_boxes.append(box)
        else:
            mirrored_boxes = boxes
        return images, mirrored_boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)

class PhotometricDistort_video(object):
    def __init__(self):
        self.pd = [
            RandomContrast_video(),
            ConvertColor_video(transform='HSV'),
            RandomSaturation_video(),
            RandomHue_video(),
            ConvertColor_video(current='HSV', transform='BGR'),
            RandomContrast_video()
        ]
        self.rand_brightness = RandomBrightness_video()
        self.rand_light_noise = RandomLightingNoise_video()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


class RandomReverse_video(object):
    """ Randomly reverse the sequence of the video sequences. """
    def __call__(self, images, boxes=None, labels=None):
        """ images is a list of images. """
        imgs = images[::-1] if random.randint(2) else images
        return imgs, boxes, labels


class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class SSDAugmentation_Bus(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class SSDAugmentationVideo(object):
    def __init__(self, size=300, mean=(104, 117, 123), expand_factor=None, crop_factor=None):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts_video(),
            ToAbsoluteCoords_video(),
            PhotometricDistort_video(),
            Expand_video(self.mean, expand_factor),
            RandomSampleCrop_video(crop_factor),
            RandomMirror_video(),
            ToPercentCoords_video(),
            Resize_video(self.size),
            SubtractMeans_video(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        """ Input to SSDAugmentationVideo are lists of images and boxes and labels."""
        return self.augment(img, boxes, labels)


class Mixup(object):
    """ Perform mixup between input images.
    The combination weights are drawn randomly from Beta distribution."""

    def __init__(self, num_mixup=2, weight='random', step=None):
        self.num_mixup = num_mixup
        self.weight = weight

    def __call__(self, images, boxes=None, labels=None):
        keyImg = images[len(images)//2].copy()
        if self._do_mixup() and random.randint(3) < 1:
            for _ in range(self.num_mixup-1):
                choice = random.randint(len(images))
                weight = self._random_weights()
                keyImg = weight * keyImg + (1 - weight) * images[choice]
        return keyImg, boxes, labels

    def _random_weights(self, alpha=0.2):
        """ Generate random weights from Beta distribution for current key image.
        We enforce the weight to be greater than 0.5 to ensure that the original
        image is largely retained. """
        if self.weight != 'random':
            return self.weight
        beta = random.beta(alpha, alpha)
        if beta < 0.5:
            beta = 1 - beta
        return beta

    def _do_mixup(self):
        return True


class SSDAugmentationVideoMixup(object):
    """ Mixup applied before actual augmentation. """

    def __init__(self, size=300, mean=(104, 117, 123), num_mixup=2, weight='random', step=None):
        """ Args:
            size: size of each input image.
            mean: image mean value to subtract.
            max_gap: maximum gap between two images to mixup.
            num_mixup: number of images to mixup (default: 2).
            mixup_scheduler: a scheduler that decide whether at current
                condition should perform mixup."""
        self.mean = mean
        self.size = size
        self.augment = Compose([
            Mixup(num_mixup, weight, step),
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        """ img is a list of numpy images. """
        return self.augment(img, boxes, labels)
