import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2_512, v2, detection_collate, VOCroot
from data import VIDDetection, VIDroot, VID_CLASSES, BaseTransform
from utils.augmentations import SSDAugmentation
# from ssd import build_ssd
from ssd_conf_shared import build_ssd_conf_shared
import numpy as np
import time
from data.scripts.create_train_lists import train_list_by_class
from layers.modules.drae import SSDReconstructionConfShared
from layers.modules.drae_loss import DRAELoss
import pickle
from collections import OrderedDict
from pprint import pprint

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Autoencoder Training')
parser.add_argument('--size', default=512, type=int, help='use 300 or 512 as base size')
parser.add_argument('--basenet', default='weights/vid_ssd_v2_512_conf_shared_batchsize_16_iter_120000_map_66.07.pth', help='pretrained base SSD model')
parser.add_argument('--version', default='v2_512', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs for each video sequence')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--lamb', default=1, type=float, help='Regularity weight for discriminative reconstruction loss')
parser.add_argument('--snapshot', default=10000, type=int, help='Snapshot interval, nonpositive values mean dont snapshot')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--class_name', default='tiger', help='Class to choose to train the autoencoder')
args = parser.parse_args()
pprint(args)

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.benchmark = True
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    print('Not using CUDA now!')

cfg = v2_512 if args.version == 'v2_512' else v2

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

train_sets = [train_list_by_class(args.class_name)]
ssd_dim = args.size
means = (104, 117, 123)  # only support voc now
num_classes = len(VID_CLASSES) + 1
batch_size = args.batch_size
max_iter = args.iterations
weight_decay = args.weight_decay
momentum = args.momentum
abnormal_frame_loss = 1e4

ssd_net = build_ssd_conf_shared('train', ssd_dim, num_classes)
ae_net = SSDReconstructionConfShared(ssd_net, cfg, args.jaccard_threshold, p=0.1)
net = ae_net


if args.cuda:
    ae_net = ae_net.cuda()

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ae_net.load_state_dict(torch.load(
        args.resume, map_location=lambda storage, loc: storage
    ))
else:
    assert args.basenet
    print('Loading SSD network {}'.format(args.basenet))
    ssd_net.load_weights(args.basenet)
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            # init.xavier_uniform(m.weight.data)
            init.normal(m.weight.data, std=0.01)
            m.bias.data.zero_()
    print('Initializing autoencoder weights...')
    ae_net.ae.apply(weights_init)

optimizer = optim.Adam(ae_net.ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = DRAELoss(lamb=args.lamb, size_average=True)
# transform = SSDAugmentation(ssd_dim, means)
transform = BaseTransform(ssd_dim, means)


def filter_roidb_by_class(roidb, class_name):
    """ For each entry in roidb, remove from annotations that is different from class_name.
    If no annotations match class_name, that entry is removed. """
    assert class_name in VID_CLASSES
    for idx, name in enumerate(VID_CLASSES):
        if name == class_name:
            class_index = idx + 1
            break

    def filter_entry_by_class(entry):
        if entry['boxes'].shape[0] == 0:
            return None
        mask = (entry['gt_classes'] == class_index)
        if not any(mask):
            return None
        entry['boxes'] = entry['boxes'][mask, :]
        entry['gt_classes'] = entry['gt_classes'][mask]
        entry['gt_overlaps'] = entry['gt_overlaps'][mask, :]
        entry['max_classes'] = entry['max_classes'][mask]
        entry['max_overlaps'] = entry['max_overlaps'][mask]
        return entry

    num_before = len(roidb)
    roidb = [filter_entry_by_class(entry) for entry in roidb]
    roidb = [entry for entry in roidb if entry is not None]
    num_after = len(roidb)
    print('filtered {} roidb entries for class {}: {} -> {}'.format(
        num_before-num_after, args.class_name, num_before, num_after
    ))
    return roidb


def filter_boxes_by_class(roidb, class_name):
    """ For each entry in roidb, remove from annotations that is different from class_name,
    but keep the entry not removed, which merely has an empty target. """
    assert class_name in VID_CLASSES
    for idx, name in enumerate(VID_CLASSES):
        if name == class_name:
            class_index = idx + 1
            break

    def filter_box(entry):
        if entry['boxes'].shape[0] == 0:
            return entry
        mask = (entry['gt_classes'] == class_index)
        entry['boxes'] = entry['boxes'][mask, :]
        entry['gt_classes'] = entry['gt_classes'][mask]
        entry['gt_overlaps'] = entry['gt_overlaps'][mask, :]
        entry['max_classes'] = entry['max_classes'][mask]
        entry['max_overlaps'] = entry['max_overlaps'][mask]
        return entry

    return [filter_box(entry) for entry in roidb]


def train():
    net.train()
    dataset = VIDDetection(train_sets, 'data/', VIDroot, transform=transform,
                           custom_filter=lambda roidb: filter_roidb_by_class(roidb, args.class_name))
    print('Training SSD on', dataset.name)

    timers = {'image': Timer(), 'net': Timer(), 'loss': Timer(), 'backward': Timer(), 'total': Timer()}
    iteration = 0
    for e in range(args.epochs):
        losses = []
        print('Training {}/{} epochs'.format(e+1, args.epochs))
        data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                      shuffle=True, collate_fn=detection_collate, pin_memory=True)
        for images, targets in data_loader:
            iteration += 1
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]
            timers['net'].tic()
            orig, recon = net(images, targets)
            t_net = timers['net'].toc(average=False)
            timers['backward'].tic()
            optimizer.zero_grad()
            loss = criterion(recon, orig)
            losses.append(loss.data[0])
            loss.backward()
            optimizer.step()
            t_backward = timers['backward'].toc(average=False)
            if iteration % 1 == 0:
                print('Iter: {:d}, net: {:.4f}s, backward: {:.4f}s, total_loss: {:.4f}'.format(
                    iteration, t_net, t_backward, loss.data[0]
                ))
            if args.snapshot > 0 and iteration % args.snapshot == 0:
                filename = 'weights/ssd{}_vid_ae_{}_{}.pth'.format(args.size, args.class_name, iteration)
                print('Saving state to {}'.format(filename))
                torch.save(ae_net.state_dict(), filename)
        print('Epoch {}/{}, average loss: {:.4f}'.format(e+1, args.epochs, np.mean(losses)))

    if args.snapshot > 0:
        final_name = os.path.join(args.save_folder, 'vid_ssd_ae_{}_{}.pth'.format(args.class_name, args.version))
        print('Saving final model {}'.format(final_name))
        torch.save(ae_net.state_dict(), final_name)


def score_frame(image, target):
    # Compute the reconstruction loss for one single frame.
    # Compute the reconstruction loss of each matched box on each frame.
    # Return a list containing scores on each frame.
    orig, recon = net(image, target)
    if len(orig) > 0:
        return torch.pow(recon.data-orig.data, 2).sum() / orig.size(0)
    else:
        return abnormal_frame_loss

def score():
    net.eval()  # disable dropout
    dataset = VIDDetection(train_sets, 'data/', VIDroot, transform=BaseTransform(ssd_dim, means), is_test=True,
                           custom_filter=lambda roidb: filter_boxes_by_class(roidb, args.class_name))
    image_set_index = dataset.imdb.image_set_index
    assert len(dataset) == len(image_set_index)
    all_losses = OrderedDict()
    timer = Timer()
    for i, (images, targets) in enumerate(dataset):
        key = image_set_index[i]
        if len(targets) == 0:
            # deal with frames not containing any object
            all_losses[key] = abnormal_frame_loss
            continue
        images.unsqueeze_(0)
        targets = [torch.FloatTensor(targets)]
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        timer.tic()
        loss = score_frame(images, targets)
        all_losses[key] = loss
        print('{:d}/{:d} ({:s}), loss: {:.4f}, time: {:.3f}s'.format(
            i+1, len(dataset), key, loss, timer.toc(average=False)
        ))

    save_filename = 'data/cache/reconstruction_loss_{}.pkl'.format(args.class_name)
    print('Saving losses results to {}'.format(save_filename))
    with open(save_filename, 'wb') as f:
        pickle.dump(all_losses, f, pickle.HIGHEST_PROTOCOL)
    save_txt = 'data/cache/results/reconstruction_loss_{}.txt'.format(args.class_name)
    print('Saving losses results to {}'.format(save_txt))
    with open(save_txt, 'wt') as f:
        for k, v in all_losses.items():
            f.write('{:s} {:.4f}\n'.format(k, v))

    return all_losses


if __name__ == '__main__':
    train()
    all_losses = score()
    sorted_losses = sorted(all_losses.items(), key=lambda x: x[1])
    saved_text = 'data/cache/results/reconstruction_loss_{}_sorted.txt'.format(args.class_name)
    print('Saving losses results to {}'.format(saved_text))
    with open(saved_text, 'wt') as f:
        for name, loss in sorted_losses:
            f.write('{:s} {:.1f}\n'.format(name, loss))
