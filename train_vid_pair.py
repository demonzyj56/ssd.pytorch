import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2_512, v2, detection_collate_pair, VOCroot
from data import VIDDetection, VIDPairDetection, VIDroot, VID_CLASSES, BaseTransform
from utils.augmentations import SSDAugmentation
# from layers.modules import MultiBoxLoss
from layers.modules.multibox_pair_loss import MultiBoxPairLoss
from ssd import build_ssd
import numpy as np
import time
from pprint import pprint

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--size', default=512, type=int, help='use 300 or 512 as base size')
parser.add_argument('--version', default='v2_512', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size of pairs of samples for training')
parser.add_argument('--resume', default='weights/vid_ssd_v2_512_mAP_65.2.pth', type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
parser.add_argument('--K', default=3, type=int, help='The size of video context')
parser.add_argument('--pair_weight', default=1, type=float, help='Hyperparameter for pair loss weight')
args = parser.parse_args()
pprint(args)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

#  cfg = (v1, v2)[args.version == 'v2']
cfg = v2_512 if args.version == 'v2_512' else v2

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# train_sets = ['DET_train_30classes','VID_train_15frames']
# train_sets = ['DET_train_30classes','VID_train_random_45frames_20180112_131510']
# train_sets = ['DET_train_30classes','VID_train_random_45frames_20180112_152248']
# train_sets = ['DET_train_30classes','VID_train_random_30frames_20180109_135846']
# train_sets = ['DET_train_30classes','VID_train_random_15frames_20180107_193804']
train_sets = ['DET_train_30classes', 'VID_train_frames']
ssd_dim = args.size
means = (104, 117, 123)  # only support voc now
num_classes = len(VID_CLASSES) + 1
batch_size = args.batch_size
accum_batch_size = 32
iter_size = accum_batch_size / batch_size
max_iter = args.iterations
weight_decay = 0.0005
# stepvalues = (160000, 200000, 240000)
# stepvalues = (200000, 250000, 300000)
# stepvalues = (int(args.iterations/3*2), int(args.iterations/6*5), args.iterations)
stepvalues = (80000, 100000, 120000)
gamma = args.gamma
momentum = args.momentum

if args.visdom:
    import visdom
    viz = visdom.Visdom()

ssd_net = build_ssd('train', ssd_dim, num_classes)
# ssd_net = build_ssd_conf_shared('train', ssd_dim, num_classes)
net = ssd_net

if args.cuda:
    #  net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True
else:
    print('not using cuda now!')


if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
    vgg_weights = torch.load(args.save_folder + args.basenet)
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)

if args.cuda:
    net = net.cuda()


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if not args.resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxPairLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)


def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...')

    # dataset = VOCDetection(args.voc_root, train_sets, SSDAugmentation(
    #     ssd_dim, means), AnnotationTransform())
    # dataset = VIDDetection(train_sets, 'data/', VIDroot, transform=SSDAugmentation(
    #     ssd_dim, means
    # ))
    dataset = VIDPairDetection(train_sets, 'data/', VIDroot, transform=BaseTransform(
        ssd_dim, means
    ), dataset_name='ImageNetVID', is_test=False, k=args.K)

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on', dataset.name)
    step_index = 0
    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
        epoch_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Epoch SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate_pair, pin_memory=True)
    for iteration in range(args.start_iter, max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if iteration in stepvalues:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * epoch,
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu() / epoch_size,
                    win=epoch_lot,
                    update='append'
                )
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
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
        loss_l, loss_c, loss_p = criterion(out, targets)
        loss = loss_l + loss_c + loss_p * args.pair_weight
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        if iteration % 10 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f || Pair loss: %.4f' % (loss.data[0], loss_p.data[0]), end=' ')
            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                viz.image(images.data[random_batch_index].cpu().numpy())
        if args.visdom:
            viz.line(
                X=torch.ones((1, 3)).cpu() * iteration,
                Y=torch.Tensor([loss_l.data[0], loss_c.data[0],
                    loss_l.data[0] + loss_c.data[0]]).unsqueeze(0).cpu(),
                win=lot,
                update='append'
            )
            # hacky fencepost solution for 0th epoch plot
            if iteration == 0:
                viz.line(
                    X=torch.zeros((1, 3)).cpu(),
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu(),
                    win=epoch_lot,
                    update=True
                )
        if iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd{}_vid_pair_{}.pth'.format(
                args.size, iteration
            ))
    torch.save(ssd_net.state_dict(), args.save_folder + 'vid_ssd_pair' + args.version + '.pth')


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
