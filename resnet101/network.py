""" ResNet101-SSD network structure.
Modified from:
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
from layers import *


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        assert dilation in (1, 2)
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=(1, 2)[dilation > 1], bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetAtrous(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetAtrous, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        s = x
        x = self.layer3(x)
        x = self.layer4(x)

        return s, x


def resnet101_atrous_reduced(pretrained=False):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetAtrous(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def prediction_module(inplanes):
    """ Prediction module for each source layer in resnet101. """
    downsample = nn.Sequential(
        nn.Conv2d(inplanes, 1024, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(1024)
    )
    return Bottleneck(inplanes=inplanes, planes=256, downsample=downsample)


def extra_block_bottleneck(inchannels, outchannels, stride):
    downsample = nn.Sequential(
        nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outchannels)
    )
    return Bottleneck(inplanes=inchannels, planes=outchannels//Bottleneck.expansion,
                      stride=stride, downsample=downsample)


resnet101_config = {
    320: {
        'mbox': [4, 6, 6, 6, 4, 4],
        'feature_maps': [40, 20, 10, 5, 3, 1],
        'min_dim': 320,
        'steps': [],
        'min_sizes': [],
        'max_sizes': [],
        'aspect_ratios': [],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'ssd320_resnet101'
    },
    512: {
        'mbox': [4, 6, 6, 6, 6, 4, 4],
        'feature_maps': [64, 32, 16, 8, 4, 2, 1],
        'min_dim': 512,
        'steps': [8, 16, 32, 64, 128, 256, 512],
        'min_sizes': [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],
        'max_sizes': [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],

        'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

        'variance' : [0.1, 0.2],

        'clip' : True,

        'name' : 'ssd512_resnet101',
    }
}


class SSDResNet101(nn.Module):
    """ Resnet101 version of SSD network. """

    def __init__(self, phase, num_classes, size=512,
                 top_k=200, conf_thresh=0.01, nms_thresh=0.45):
        assert size == 512, "Only support input size 512 for now"
        super(SSDResNet101, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size
        self.cfg = resnet101_config[size]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)

        self.resnet101 = resnet101_atrous_reduced(pretrained=True)
        self.extras = nn.ModuleList([
            extra_block_bottleneck(x, 1024, s) for x, s in \
                zip([2048, 1024, 1024, 1024, 1024], [2, 2, 2, 2, 1])
        ])
        self.prediction_module = nn.ModuleList([
            prediction_module(x) for x in [512, 2048, 1024, 1024, 1024, 1024, 1024]
        ])
        self.loc = nn.ModuleList([
            nn.Conv2d(1024, x*4, kernel_size=3, padding=1) for x in self.cfg['mbox']
        ])
        self.conf = nn.ModuleList([
            nn.Conv2d(1024, x*num_classes, kernel_size=3, padding=1) for x in self.cfg['mbox']
        ])

        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, top_k, conf_thresh, nms_thresh)

    def forward(self, x):
        loc = list()
        conf = list()
        sources = self.sources(x)

        # apply multibox head to source layers
        for (x, p, l, c) in zip(sources, self.prediction_module, self.loc, self.conf):
            px = p(x)
            loc.append(l(px).permute(0, 2, 3, 1).contiguous())
            conf.append(c(px).permute(0, 2, 3, 1).contiguous())
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

    def sources(self, x):
        sources = list(self.resnet101(x))
        for extra in self.extras:
            sources.append(extra(sources[-1]))
        return sources

    def load_weights(self, base_file):
        import os
        assert os.path.exists(base_file), "Model file {} does not exist".format(base_file)
        assert base_file.endswith('.pth'), "Only .pth files are supported"
        print('Loading weights into state dict...')
        self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
        print('Finished!')


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    a = Variable(torch.randn(1, 3, 512, 512))
    net = SSDResNet101('train', 21)
    b = net(a)
    from IPython import embed; embed()
