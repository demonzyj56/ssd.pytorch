""" SSD with VGG16 backbone and FPN head. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import PriorBox, L2Norm, Detect
from data import v2, v2_512
import os


class SSDFPN(nn.Module):
    """Single Shot Multibox Architecture with feature pyramid network.
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    In specific, the FPN structure is only attached to the first three
    layers, normalizing them to 256 channels.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to fpn
        fpn: FPN structure
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, fpn, loc_head, conf_head,
                 num_classes, cfg, sz=300):
        super(SSDFPN, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.priorbox = PriorBox(v2 if sz == 300 else v2_512)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = sz
        self.cfg = cfg

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.fpn = fpn
        self.loc = loc_head
        self.conf = conf_head

        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def sources(self, x):
        """ Calculates the feature pyramid for loc and conf. """
        sources = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        fp = self.fpn(sources[:3]) + sources[3:]
        return fp

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
        loc = list()
        conf = list()
        sources = self.sources(x)

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
            self.load_state_dict(torch.load(base_file, \
                map_location=lambda storage, loc: storage), strict=False)
            print('Finished loading {}'.format(base_file))
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag],
                                     stride=2, padding=1)]
            elif v == '512Last':
                layers += [nn.Conv2d(in_channels, 256, kernel_size=4,
                                     padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def fcn_subnet(in_channels, mid_channels, out_channels):
    """ Build a 2-layer fcn subnet for confidence/location. """
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
    )


def multibox_head_shared(vgg_layers, extra_layers, cfg, num_classes):
    """ We share the confidence and location with each source layer.
    This requires that the input dimension should be identical. """
    source_layers = [vgg_layers[24], vgg_layers[-2]]
    source_layers.append(extra_layers[1])
    # all nn.Conv2d
    source_channels = [s.out_channels for s in source_layers]
    fpn = FPN(source_channels)
    conf_layers = fcn_subnet(256, 256, max(cfg)*num_classes)
    loc_layers = fcn_subnet(256, 256, max(cfg)*4)

    return fpn, loc_layers, conf_layers


def multibox_head_unshared(vgg_layers, extra_layers, cfg, num_classes):
    """ The confidence and location classifiers are not shared. """
    source_layers = [vgg_layers[24], vgg_layers[-2]]
    source_layers += extra_layers[1::2]
    # all nn.Conv2d
    source_channels = [s.out_channels for s in source_layers]
    fpn = FPN(source_channels[:3])
    conf_layers = nn.ModuleList([
        nn.Conv2d(256, m*num_classes, kernel_size=3, padding=1) for m in cfg
    ])
    loc_layers = nn.ModuleList([
        nn.Conv2d(256, m*4, kernel_size=3, padding=1) for m in cfg
    ])

    return fpn, loc_layers, conf_layers


class FPN(nn.Module):
    """ The feature pyramid network on top of the given SSD module.
    For the given source channels, construct two sets of weights: the
    first set is 1x1 conv to reduce the dimention to 256, and the second
    set is 3x3 conv on the merged feature map. """

    def __init__(self, source_channels, out_channels=256):
        super(FPN, self).__init__()
        self.conv1x1 = nn.ModuleList([
            nn.Conv2d(s, out_channels, kernel_size=1) for s in source_channels
        ])
        # The first one does not require merging
        self.conv3x3 = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1) for _ in source_channels[:-1]
        ])

    def forward(self, sources):
        """ Sources are the source layers the decrease the dimentionality
        sequentially.
        The returned sources are bottom-up as well."""
        merged = []
        for s, conv1x1 in zip(reversed(sources), reversed(self.conv1x1)):
            source_merged = conv1x1(s)
            if len(merged) > 0:
                source_merged += F.upsample(
                    merged[-1], scale_factor=2, mode='bilinear'
                )
            merged.append(source_merged)
        for idx, conv3x3 in enumerate(self.conv3x3):
            merged[idx+1] = conv3x3(merged[idx+1])

        return merged[::-1]


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128,
            '512Last'],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
}


def build_ssd_fpn(phase, size=300, num_classes=21):
    """ Builder for SSD with FPN head. """
    assert phase in ('train', 'test')

    vgg_layers = vgg(base[str(size)], 3)
    extra_layers = add_extras(extras[str(size)], 1024)
    fpn, loc, conf = multibox_head_unshared(vgg_layers, extra_layers,
                                            mbox[str(size)], num_classes)

    return SSDFPN(phase, vgg_layers, extra_layers, fpn, loc, conf,
                  num_classes=num_classes, cfg=mbox[str(size)], sz=size)
