""" Analyze how the validation set is composed of. """
from data import VID_CLASSES, VIDroot
from vid.dataset.imagenet_vid import ImageNetVID


def get_class_dist(gt_roidb):
    """ Given a gt_roidb, counts number of gt bbox for each class.
    Returns a list where each location gives how many boxes falls on that class. """
    # d = dict(zip(VID_CLASSES, [0 for _ in VID_CLASSES]))
    d = [0 for _ in VID_CLASSES]
    for roi_rec in gt_roidb:
        for cls_idx in roi_rec['gt_classes']:
            d[cls_idx-1] += 1
    return d


def generate_val_frame_list_by_video():
    """For the ith_video in the VID_val_videos list, generate the corresponding
    video list."""
    with open('data/VID_val_videos.txt', 'r') as f:
        lines = [x.strip().split(' ') for x in f.readlines()]
    video_basenames = [x[0] for x in lines]
    num_frames = [int(x[-1]) for x in lines]
    for i, video_name in enumerate(video_basenames):
        with open('data/VID_val_frames_seq_{}.txt'.format(i+1), 'wt') as f:
            for j in range(num_frames[i]):
                f.write("{:s}/{:06d} {:d}\n".format(video_name, j, j+1))
    print("Total number of frames: {}".format(sum(num_frames)))


def generate_val_list(path):
    """ Generate validation list similar to train_xxx.txt.
    Each file has name val_{idx}.txt where idx is the class index. """
    val_list = [[] for _ in VID_CLASSES]
    with open('data/VID_val_videos.txt', 'r') as f:
        lines = [x.strip().split(' ') for x in f.readlines()]
    video_basenames = [x[0] for x in lines]
    num_frames = [int(x[-1]) for x in lines]
    num_frames_index = dict(zip(video_basenames, num_frames))
    for i, video_name in enumerate(video_basenames):
        imdb = ImageNetVID('VID_val_frames_seq_{}'.format(i+1),
                           'data/', VIDroot)
        class_dist = get_class_dist(imdb.gt_roidb())
        for j, num_boxes in enumerate(class_dist):
            if num_boxes > 0:
                val_list[j].append(video_name)
    for cls_idx in range(len(VID_CLASSES)):
        print("Processing {}/{} categories".format(cls_idx+1, len(VID_CLASSES)))
        cls_video_list = val_list[cls_idx]
        with open(path+'VID_val_frames_class_{}.txt'.format(VID_CLASSES[cls_idx]), 'wt') as f:
            for video_name in cls_video_list:
                frames = num_frames_index[video_name]
                for j in range(frames):
                    f.write("{:s}/{:06d} {:d}\n".format(video_name, j, j+1))


if __name__ == "__main__":
    # generate_val_frame_list_by_video()
    # generate_val_list('data/')
    from data import VIDVideoDetection, BaseTransform_video
    dataset_mean = (104, 117, 123)
    dataset = VIDVideoDetection(['DET_train_30classes','VID_train_K_1'], 'data/', VIDroot,
                                transform=BaseTransform_video(512, dataset_mean),
                                is_test=False, k=0)
    class_list = get_class_dist(dataset.gt_roidb)
    # print(dict(zip(VID_CLASSES, class_list)))
    for ix, class_name in enumerate(VID_CLASSES):
        print("{}: {}".format(class_name, class_list[ix]))