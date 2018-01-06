""" Create various training txt files for VID. """
import os
import numpy as np
from datetime import datetime
from data import VIDroot


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


if __name__ == '__main__':
    random_train_frames(15)