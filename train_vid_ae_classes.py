import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2_512, v2, detection_collate, VOCroot
from data import VIDDetection, VIDroot, VID_CLASSES, BaseTransform
from utils.augmentations import SSDAugmentation
from layers.modules import SSDReconstruction, DiscriminativeReconstructionLoss
from ssd import build_ssd
import numpy as np
import time
from data.scripts.create_train_lists import train_list_by_class

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Autoencoder Training')
parser.add_argument('--size', default=512, type=int, help='use 300 or 512 as base size')
parser.add_argument('--basenet', default='weights/vid_ssd_v2_512_mAP_65.2.pth', help='pretrained base SSD model')
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
parser.add_argument('--class_name', default='lion', help='Class to choose to train the autoencoder')
args = parser.parse_args()


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

ssd_net = build_ssd('train', ssd_dim, num_classes)
ae_net = SSDReconstruction(ssd_net, p=0.1)
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

# optimizer = optim.SGD(ae_net.ae.parameters(), lr=args.lr,
#                       momentum=args.momentum, weight_decay=args.weight_decay)
optimizer = optim.Adam(ae_net.ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = DiscriminativeReconstructionLoss(args.lamb, cfg, 0.5)
# transform = SSDAugmentation(ssd_dim, means)
transform = BaseTransform(ssd_dim, means)


def filter_roidb_by_class(roidb, class_name):
    return roidb


def train():
    net.train()
    dataset = VIDDetection(train_sets, 'data/', VIDroot, transform=transform,
                           custom_filter=lambda roidb: filter_roidb_by_class(roidb, args.class_name))
    print('Training SSD on', dataset.name)

    timers = {'image': Timer(), 'net': Timer(), 'loss': Timer(), 'backward': Timer(), 'total': Timer()}
    iteration = 0
    # all_features = [[] for _ in range(7)]
    for e in range(args.epochs):
        epoch_loss = [0 for _ in range(7)]
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
            out = net(images)
            t_net = timers['net'].toc(average=False)
            timers['loss'].tic()
            losses = criterion(out, targets)
            t_loss = timers['loss'].toc(average=False)
            # # ================================================
            # feature_maps = net.ssd_feature_maps(images)
            # features = criterion.samples(feature_maps, targets)
            # for idx, f in enumerate(features):
            #     if len(f) > 0:
            #         all_features[idx].append(f)
            # print('Sample {}/{}'.format(iteration, len(dataset)))
            # # ================================================
            timers['backward'].tic()
            for idx, l in enumerate(losses):
                epoch_loss[idx] += l.data[0] * args.batch_size
                if l.requires_grad:
                    l.backward()
            optimizer.step()
            t_backward = timers['backward'].toc(average=False)
            if iteration % 1 == 0:
                loss_fmt = ', '.join(['{:.1f}'.format(l.data[0]) for l in losses])
                print('Iter: {:d}, net: {:.4f}s, match: {:.4f}s, backward: {:.4f}s, loss: {:s}'.format(
                    iteration, t_net, t_loss, t_backward, loss_fmt
                ))
            if args.snapshot > 0 and iteration % args.snapshot == 0:
                filename = 'weights/ssd{}_vid_ae_{}.pth'.format(args.size, iteration)
                print('Saving state to {}'.format(filename))
                torch.save(ae_net.state_dict(), filename)

        print('Finish epoch {:d}/{:d}, final average loss: '.format(e+1, args.epochs))
        print([el/len(dataset) for el in epoch_loss])

    final_name = os.path.join(args.save_folder, 'vid_ssd_ae_{}.pth'.format(args.version))
    print('Saving final model {}'.format(final_name))
    torch.save(ae_net.state_dict(), final_name)
    # # ==================================================
    # print('Saving cached features...')
    # for i in range(7):
    #     if len(all_features[i]) > 0:
    #         f = np.concatenate(all_features[i], axis=0)
    #         np.save('data/cache/features_{}_{}'.format(train_sets[0], i), f)
    # # ==================================================

if __name__ == '__main__':
    train()


