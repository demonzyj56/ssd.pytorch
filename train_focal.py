import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2_512, v2, v1, AnnotationTransform, VOCDetection, detection_collate, VOCroot, VOC_CLASSES
from utils.augmentations import SSDAugmentation
from layers.modules.multibox_focal_loss import MultiBoxFocalLoss
from ssd_fpn import build_ssd_fpn
import numpy as np
import time
from pprint import pprint

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--size', default=512, type=int, help='use 300 or 512 as base size')
parser.add_argument('--version', default=v2_512['name'], help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold_hi', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--jaccard_threshold_lo', default=0.4, type=float, help='Low threshold for negative samples')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--multigpu', default=False, type=str2bool, help='Use multigpu setting')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
parser.add_argument('--alpha_focal', default=0.25, type=float, help='Multiplier for focal loss')
parser.add_argument('--gamma_focal', default=2, type=float, help='Power for focal loss')
parser.add_argument('--warm_up', default=0, type=int, help='Warm up iterations before training')
args = parser.parse_args()
print('Args:')
pprint(args)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

#  cfg = (v1, v2)[args.version == 'v2']
cfg = v2_512 if args.version == 'v2_512' else v2

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
# train_sets = 'train'
ssd_dim = args.size
means = (104, 117, 123)  # only support voc now
num_classes = len(VOC_CLASSES) + 1
batch_size = args.batch_size
accum_batch_size = 32
iter_size = accum_batch_size / batch_size
max_iter = 120000
weight_decay = 0.0005
stepvalues = (80000, 100000, 120000)
gamma = 0.1
momentum = 0.9

if args.visdom:
    import visdom
    viz = visdom.Visdom()

ssd_net = build_ssd_fpn('train', ssd_dim, num_classes)
net = ssd_net

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def final_bias_init(m):
    bias = np.zeros((num_classes, ), dtype=np.float32)
    bias[0] = np.log((num_classes-1) * (1-0.01) / 0.01)
    bias = np.vstack([bias for _ in range(6)])
    m.bias.data.copy_(torch.Tensor(bias))

if args.cuda:
    cudnn.benchmark = True
    net = net.cuda()
    if args.multigpu:
        net = torch.nn.DataParallel(net)

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
    print('Initializing weights...')
    ssd_net.apply(weights_init)
    vgg_weights = torch.load(args.save_folder + args.basenet)
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)
    print('Initializing final bias...')
    final_bias_init(ssd_net.conf[-1])

if args.cuda:
    net = net.cuda()


optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxFocalLoss(num_classes, args.jaccard_threshold_hi,
                              args.jaccard_threshold_lo,
                              alpha=args.alpha_focal, gamma=args.gamma_focal)

def train():
    net.train()
    # loss counters
    epoch = 0
    print('Loading Dataset...')

    dataset = VOCDetection(args.voc_root, train_sets, SSDAugmentation(
        ssd_dim, means), AnnotationTransform())

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on', dataset.name)
    step_index = 0
    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    tic = time.time()
    warm_up = False
    for iteration in range(args.start_iter-args.warm_up, max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if iteration >= 0 and warm_up:
            print('Resume lr to {}'.format(args.lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            warm_up = False
        if iteration < 0 and not warm_up:
            print('Warming up for {:d} iterations: set lr to {}'.format(args.warm_up, args.lr/10))
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr / 10
            warm_up = True
        # if iteration < 0:
        #     warm_up_lr = args.lr / 10
        #     warm_up_iters = iteration + args.warm_up
        #     warm_up_coeff = warm_up_iters / args.warm_up
        #     cur_lr = (1-warm_up_coeff) * warm_up_lr + warm_up_coeff * args.lr
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = cur_lr
        if iteration in stepvalues:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
            epoch += 1

        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        if iteration % 10 == 0:
            print('Timer: %.4f sec' % ((time.time()-tic)/10))
            print('iter ' + repr(iteration) + ' || loc loss: %.4f, conf loss: %.4f ||' % (loss_l.data[0], loss_c.data[0]), end=' ')
            tic = time.time()
        if iteration % 10000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd{}_focal_0712_{}.pth'.format(
                args.size, iteration
            ))
    #  torch.save(ssd_net.state_dict(), args.save_folder + '' + args.version + '.pth')
    torch.save(ssd_net.state_dict(), args.save_folder + 'voc_ssd_focal_' + args.version + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
