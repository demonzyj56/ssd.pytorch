""" Create various training txt files for VID. """
import os
import numpy as np
from datetime import datetime
from data import VIDroot, VID_CLASSES


def create_train_video_list():
    """ Create a list that contains all VID training sequences. """
    train_frames_file = os.path.join(VIDroot, 'ImageSets', 'VID', 'train.txt')
    train_videos_file = os.path.join('data', 'VID_train_videos.txt')
    assert os.path.exists(train_frames_file), "{} does not exist".format(train_frames_file)
    with open(train_frames_file, 'r') as f:
        lines = [x.strip().split(' ') for x in f.readlines()]
    sequences = {}
    for ix, line in enumerate(lines):
        if (ix+1) % 10000 == 0:
            print('Processing {}/{}'.format(ix+1, len(lines)))
        if line[0].startswith('ILSVRC2017'):
            continue
        path_parts = line[0].split('/')
        data_path = 'train/{}/{}'.format(path_parts[0], path_parts[1])
        try:
            sequences[data_path] += 1
        except KeyError:
            sequences[data_path] = 1

    with open(train_videos_file, 'wt') as f:
        count = 1
        for path, num_frames in sequences.items():
            f.write('{:s} {:d} 0 {:d}\n'.format(path, count, num_frames))
            count += num_frames


def random_train_frames(num_sampled):
    """ Create a txt file under data/ which samples num_sampled samples from
    each training sequences."""
    with open('data/VID_train_videos.txt', 'r') as f:
        lines = [x.strip().split(' ') for x in f.readlines()]
    filename = "data/VID_train_random_{:d}frames_{:%Y%m%d_%H%M%S}.txt".format(
        num_sampled, datetime.now()
    )
    with open(filename, 'wt') as f:
        for line in lines:
            file_path, num_frames = line[0], int(line[-1])
            random_indices = np.random.choice(num_frames, min(num_sampled, num_frames), replace=False)
            for idx in sorted(random_indices):
                f.write('{:s} 1 {:d} {:d}\n'.format(file_path, idx, num_frames))


def random_train_file():
    with open('data/VID_train_videos.txt', 'r') as f:
        lines = [x.strip().split(' ') for x in f.readlines()]
    rand_seq = np.random.randint(len(lines))
    selected = lines[rand_seq]
    seq_number = int(selected[-1])
    seq_name = 'VID_train_seq_{}'.format(rand_seq)
    with open('data/{}.txt'.format(seq_name), 'wt') as f:
        for i in range(seq_number):
            f.write('{:s} 1 {:d} {:d}\n'.format(
                selected[0], i, seq_number
            ))
    print('Done generating {}, number of frames: {}'.format(seq_name, seq_number))
    return seq_name


def train_list_by_volume(name):
    """ Create train list by volume name,
    which is given by train/ILSVRC2015_VID_train_000x_ILSVRC2015_train_xxxxxxxx. """
    with open('data/VID_train_videos.txt', 'r') as f:
        lines = [x.strip().split(' ') for x in f.readlines()]
    name_to_frames = dict([(x[0], int(x[-1])) for x in lines])
    assert name in name_to_frames.keys()
    _, mid, end = name.split('/')
    train_list_name = 'VID_train_frames_{}_{}'.format(mid.split('_')[-1], end.split('_')[-1])
    num_frames = name_to_frames[name]
    with open('data/{}.txt'.format(train_list_name), 'wt') as f:
        for i in range(num_frames):
            f.write('{:s} 1 {:d} {:d}\n'.format(name, i, num_frames))
    print('Done generating {}, number of frames: {}'.format(train_list_name, num_frames))

    return train_list_name

def train_list_by_class(name):

    def class_name_to_id(class_name):
        assert class_name in VID_CLASSES
        for idx, name in enumerate(VID_CLASSES):
            if name == class_name:
                return idx

    id = class_name_to_id(name) + 1
    with open(os.path.join(VIDroot, 'ImageSets', 'VID', 'train_{}.txt'.format(id)), 'r') as f:
        video_names = [x.strip().split(' ')[0] for x in f.readlines()]
    video_names = ['train/{}'.format(n) for n in video_names if not n.startswith('ILSVRC2017')]
    with open(os.path.join('data', 'VID_train_videos.txt'), 'r') as f:
        lines = [x.strip().split(' ') for x in f.readlines()]
        names_to_frames = dict([(line[0], int(line[-1])) for line in lines])
    with open('data/VID_train_frames_{}.txt'.format(name), 'wt') as f:
        for video_name in video_names:
            frames = names_to_frames[video_name]
            for idx in range(frames):
                f.write('{:s} 1 {:d} {:d}\n'.format(video_name, idx, frames))

    return 'VID_train_frames_{}'.format(name)


if __name__ == '__main__':
    for class_name in VID_CLASSES:
        print('Creating {}'.format(class_name))
        train_list_by_class(class_name)