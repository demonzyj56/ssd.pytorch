import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import res101_300, VOC_CLASSES
import os
import torchvision.models as models
from .resnet_modified import resnet101_no_fc


config = res101_300


class SSD_res(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, head, num_classes, sz=300):
        super(SSD_res, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBox(config)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = sz

        # SSD network
        self.res101 = base
        # Layer learns to scale the l2 normalized features from conv4_3
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply res101 and yield outputs from layer2 and layer4
        map1, x = self.res101(x)
        sources.append(map1)
        sources.append(x)


        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            if k == len(self.extras)-1:  # last ave. pooling layer
                x = v(x)
            else:
                x = F.relu(v(x), inplace=True)
            if k % 2 == 1 or k==len(self.extras)-1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



def add_extras(i):
    # Extra layers added to VGG for feature scaling
    '''

    :param cfg:
    :param i: input channels: 2048 for res101
    :param batch_norm:
    :return:
    '''
    layers = []
    layers += [nn.Sequential(nn.Conv2d(in_channels=i,out_channels=256,padding=0,bias=False,stride=1,kernel_size=1), nn.BatchNorm2d(256))]
    layers += [nn.Sequential(nn.Conv2d(in_channels=256,out_channels=512,padding=1,bias=False,stride=2,kernel_size=3), nn.BatchNorm2d(512))]
    layers += [nn.Sequential(nn.Conv2d(in_channels=512, out_channels=256, padding=0, bias=False, stride=1, kernel_size=1), nn.BatchNorm2d(256))]
    layers += [nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, padding=1, bias=False, stride=2, kernel_size=3), nn.BatchNorm2d(512))]
    layers += [nn.Sequential(nn.Conv2d(in_channels=512, out_channels=256, padding=0, bias=False, stride=1, kernel_size=1),nn.BatchNorm2d(256))]
    layers += [nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, padding=1, bias=False, stride=2, kernel_size=3), nn.BatchNorm2d(512))]
    layers += [nn.AvgPool2d(kernel_size=2)]

    return layers


def multibox(res101, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    # res101 layers
    loc_layers += [nn.Conv2d(res101.layer2[-1].conv3.out_channels,
                                 cfg[0] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(res101.layer2[-1].conv3.out_channels,
                              cfg[0] * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(res101.layer4[-1].conv3.out_channels,
                             cfg[1] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(res101.layer4[-1].conv3.out_channels,
                              cfg[1] * num_classes, kernel_size=3, padding=1)]

    # extra layers
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v[0].out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v[0].out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    # for the final pooling layer
    loc_layers += [nn.Conv2d(extra_layers[-2][0].out_channels, cfg[k]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(extra_layers[-2][0].out_channels, cfg[k]
                              * num_classes, kernel_size=3, padding=1)]
    return res101, extra_layers, (loc_layers, conf_layers)





base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    # '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
    '300': [256, 512, 256, 512, 256, 512]
}
mbox = {
    # '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location

    '512': [],
    '300': [3, 6, 6, 6, 6, 6]
}





def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Sorry only SSD300 or SSD300_101 are supported currently!")
        return

    return SSD_res(phase, *multibox(resnet101_no_fc(),
                   add_extras(2048),
                   mbox[str(size)], num_classes), num_classes, size)


