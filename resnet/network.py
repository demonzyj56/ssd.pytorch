""" ResNet-SSD network structure.
Modified from:
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import os
import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from layers import PriorBox, Detect


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    assert dilation in (1, 2)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=(1, 2)[dilation > 1], bias=False,
                     dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        assert dilation in (1, 2)
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=(1, 2)[dilation > 1], bias=False,
                               dilation=dilation)
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


class ResNetReduced(nn.Module):
    """ Returns resnet with final avgpool and fc removed.
    Optionally, if atrous is True, then apply the atrous algorithm on the
    final residual block to reduce the stride to 1.

    Returns all sources layers (conv3 - conv5).
    """

    def __init__(self, block, layers, atrous=False):
        self.inplanes = 64
        super(ResNetReduced, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        if atrous:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2)
        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

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
        layers.append(
            block(self.inplanes, planes, stride, dilation, downsample)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        """ Returns conv3 - conv5 which have stride of (8, 16, 32). """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        conv3 = self.layer2(x)
        conv4 = self.layer3(conv3)
        conv5 = self.layer4(conv4)

        return conv3, conv4, conv5


def resnet34_reduced(pretrained=False, atrous=False, freeze_batchnorm=False):
    """ Returns ResNet-34 model. """
    model = ResNetReduced(BasicBlock, [3, 4, 6, 3], atrous)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']),
                              strict=False)
    if freeze_batchnorm:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # Freeze weight and bias in affine transformation
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                # Still compute running mean/var, but do not update
                m.momentum = 0

    return model


def resnet50_reduced(pretrained=False, atrous=False, freeze_batchnorm=False):
    """ Returns ResNet-50 model. """
    model = ResNetReduced(Bottleneck, [3, 4, 6, 3], atrous)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),
                              strict=False)
    if freeze_batchnorm:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # Freeze weight and bias in affine transformation
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                # Still compute running mean/var, but do not update
                m.momentum = 0

    return model


def resnet101_reduced(pretrained=False, atrous=False, freeze_batchnorm=False):
    """ Returns ResNet-101 model. """
    model = ResNetReduced(Bottleneck, [3, 4, 23, 3], atrous)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),
                              strict=False)
    if freeze_batchnorm:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # Freeze weight and bias in affine transformation
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                # Still compute running mean/var, but do not update
                m.momentum = 0

    return model


def extras320_simple(in_channels):
    """ Simple version of extra layers. Fails miserably. """
    # input: 10x10, output: 5x5
    block6 = nn.Sequential(OrderedDict([
        ('conv6', nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=2, bias=False)),
        ('relu6', nn.ReLU(inplace=True))
    ]))
    # input: 5x5, output: 3x3
    block7 = nn.Sequential(OrderedDict([
        ('conv7', nn.Conv2d(256, 256, kernel_size=3, bias=False)),
        ('relu7', nn.ReLU(inplace=True))
    ]))
    # input: 3x3, output: 1x1
    block8 = nn.Sequential(OrderedDict([
        ('conv8', nn.Conv2d(256, 256, kernel_size=3, bias=False)),
        ('relu8', nn.ReLU(inplace=True))
    ]))

    return nn.ModuleList([block6, block7, block8])


def extras320(in_channels=2048, bias=True):
    """ Returns extra layers when input size is 320x320. """
    # input: 10x10, output: 5x5
    block6 = nn.Sequential(OrderedDict([
        ('conv6_1', nn.Conv2d(in_channels, 128, kernel_size=1, bias=bias)),
        ('relu6_1', nn.ReLU(inplace=True)),
        ('conv6_2', nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=bias)),
        ('relu6_2', nn.ReLU(inplace=True)),
    ]))
    # input: 5x5, output: 3x3
    block7 = nn.Sequential(OrderedDict([
        ('conv7_1', nn.Conv2d(256, 128, kernel_size=1, bias=bias)),
        ('relu7_1', nn.ReLU(inplace=True)),
        ('conv7_2', nn.Conv2d(128, 256, kernel_size=3, bias=bias)),
        ('relu7_2', nn.ReLU(inplace=True)),
    ]))
    # input: 3x3, output: 1x1
    block8 = nn.Sequential(OrderedDict([
        ('conv8_1', nn.Conv2d(256, 128, kernel_size=1, bias=bias)),
        ('relu8_1', nn.ReLU(inplace=True)),
        ('conv8_2', nn.Conv2d(128, 256, kernel_size=3, bias=bias)),
        ('relu8_2', nn.ReLU(inplace=True)),
    ]))

    return nn.ModuleList([block6, block7, block8])


def extras320_residual(in_channels=2048):
    """ Return residual blocks as extra layers. """
    _downsample = lambda _in_channels, out_channels, stride: nn.Sequential(
        nn.Conv2d(_in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels)
    )
    if in_channels == 2048:
        block_func = Bottleneck
    else:
        block_func = BasicBlock
    block6 = block_func(in_channels, 256, stride=2,
                        downsample=_downsample(in_channels, 256, 2))
    block7 = block_func(256, 256, stride=2, downsample=_downsample(256, 256, 2))
    block8 = nn.AvgPool2d(3)

    return nn.ModuleList([block6, block7, block8])



def extras320_large_separable(in_channels=2048):
    """ Large separable convolution used in light-head rcnn. """
    class _LSConv(nn.Module):
        def __init__(self, _in_channels, mid_channels, out_channels,
                     kernel_size, stride, padding):
            super(_LSConv, self).__init__()
            self.left = nn.Sequential(OrderedDict([
                ('conv1_1', nn.Conv2d(_in_channels, mid_channels,
                                      kernel_size=(1, kernel_size),
                                      padding=(0, padding),
                                      stride=(1, stride), bias=False)),
                ('relu1_1', nn.ReLU(inplace=True)),
                ('conv1_2', nn.Conv2d(mid_channels, out_channels,
                                      kernel_size=(kernel_size, 1),
                                      padding=(padding, 0),
                                      stride=(stride, 1), bias=False))
            ]))
            self.right = nn.Sequential(OrderedDict([
                ('conv2_1', nn.Conv2d(_in_channels, mid_channels,
                                      kernel_size=(kernel_size, 1),
                                      padding=(padding, 0),
                                      stride=(stride, 1), bias=False)),
                ('relu2_1', nn.ReLU(inplace=True)),
                ('conv2_2', nn.Conv2d(mid_channels, out_channels,
                                      kernel_size=(1, kernel_size),
                                      padding=(0, padding),
                                      stride=(1, stride), bias=False))
            ]))
        def forward(self, x):
            left = self.left(x)
            right = self.right(x)
            return F.relu(left + right, inplace=True)

    # input: 10x10, output: 5x5
    block6 = _LSConv(in_channels, 128, 256, 7, 2, 3)
    # input: 5x5, output: 3x3
    block7 = _LSConv(256, 128, 256, 5, 2, 2)
    # input: 3x3, output: 1x1
    block8 = _LSConv(256, 128, 256, 3, 1, 0)

    return nn.ModuleList([block6, block7, block8])


def extras512(in_channels, bias=True):
    """ Returns extra layers when input size is 512x512. """
    # input: 16x16, output: 8x8
    block6 = nn.Sequential(OrderedDict([
        ('conv6_1', nn.Conv2d(in_channels, 128, kernel_size=1, bias=bias)),
        ('relu6_1', nn.ReLU(inplace=True)),
        ('conv6_2', nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=bias)),
        ('relu6_2', nn.ReLU(inplace=True)),
    ]))
    # input: 8x8, output: 4x4
    block7 = nn.Sequential(OrderedDict([
        ('conv7_1', nn.Conv2d(256, 128, kernel_size=1, bias=bias)),
        ('relu7_1', nn.ReLU(inplace=True)),
        ('conv7_2', nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2, bias=bias)),
        ('relu7_2', nn.ReLU(inplace=True)),
    ]))
    # input: 4x4, output: 2x2
    block8 = nn.Sequential(OrderedDict([
        ('conv8_1', nn.Conv2d(256, 128, kernel_size=1, bias=bias)),
        ('relu8_1', nn.ReLU(inplace=True)),
        ('conv8_2', nn.Conv2d(128, 256, kernel_size=3, bias=bias)),
        ('relu8_2', nn.ReLU(inplace=True)),
    ]))
    # input: 2x2, output: 1x1
    block9 = nn.Sequential(OrderedDict([
        ('conv9_1', nn.Conv2d(256, 128, kernel_size=1, bias=bias)),
        ('relu9_1', nn.ReLU(inplace=True)),
        ('conv9_2', nn.Conv2d(128, 256, kernel_size=2, bias=bias)),
        ('relu9_2', nn.ReLU(inplace=True)),
    ]))

    return nn.ModuleList([block6, block7, block8, block9])


class MultiboxHeadUnshared(nn.Module):
    """ One conv layer for each source as a classifier/regressor.
    The forward function returns all decision values.
    This abstracts away the difference between shared/unshared head. """

    def __init__(self, in_channels, mbox, multiplier):
        super(MultiboxHeadUnshared, self).__init__()
        self.decision_layers = nn.ModuleList([
            nn.Conv2d(c, m*multiplier, kernel_size=3, padding=1)
            for c, m in zip(in_channels, mbox)
        ])

    def forward(self, sources):
        return [dl(s) for dl, s in zip(self.decision_layers, sources)]


def extra_func_dispatch(size, extra_config):
    """ Dispatch to different extra layers. """
    if size == 320:
        if extra_config == 'simple':
            return extras320_simple
        elif extra_config == 'normal':
            return extras320
        elif extra_config == 'large_separable':
            return extras320_large_separable
        elif extra_config == 'residual':
            return extras320_residual
        else:
            return NotImplemented
    elif size == 512:
        if extra_config == 'normal':
            return extras512
        else:
            return NotImplemented
    else:
        return NotImplemented


def multibox(size, num_classes, source_channels, extra_config):
    """ Factory that returns all layers for SSD except base ResNet network.
    Everything is hard-coded here. """
    extras = extra_func_dispatch(size, extra_config)(source_channels[-1])
    if size == 320:
        mbox = [4, 6, 6, 6, 4, 4]
        cfg = {
            'feature_maps': [40, 20, 10, 5, 3, 1],
            'min_dim': 320,
            'steps': [8, 16, 32, 64, 106.7, 320],
            'min_sizes': [22.4, 48.0, 105.6, 163.2, 220.8, 278.4],
            'max_sizes': [48.0, 105.6, 163.2, 220.8, 278.4, 336.0],
            #  'min_sizes': [32.0, 64.0, 118.4, 172.8, 227.2, 281.6],
            #  'max_sizes': [64.0, 118.4, 172.8, 227.2, 281.6, 336.0],
            'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            'variance' : [0.1, 0.2],
            'clip' : True,
            'name' : 'resnet320',
        }
    elif size == 512:
        mbox = [4, 6, 6, 6, 6, 4, 4]
        cfg = {
            'feature_maps': [64, 32, 16, 8, 4, 2, 1],
            'min_dim': 512,
            'steps': [8, 16, 32, 64, 128, 256, 512],
            'min_sizes': [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],
            'max_sizes': [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
            'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
            'variance' : [0.1, 0.2],
            'clip' : True,
            'name' : 'resnet512',
        }
    else:
        raise ValueError('Size {:d} is not supported'.format(size))
    for _ in extras:
        # hard coded here
        source_channels.append(256)
    loc_layers = MultiboxHeadUnshared(source_channels, mbox, 4)
    conf_layers = MultiboxHeadUnshared(source_channels, mbox, num_classes)

    for block in (extras, loc_layers, conf_layers):
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
                try:
                    m.bias.data.zero_()
                except:
                    pass
            if isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()

    return cfg, extras, loc_layers, conf_layers


class SSD(nn.Module):
    """ ResNet version of SSD network. """

    def __init__(self, phase, cfg, base, extras, loc, conf, num_classes, size):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size
        self.cfg = cfg
        self.priors = Variable(PriorBox(self.cfg).forward(), volatile=True)
        self.resnet = base
        self.extras = extras
        self.loc = loc
        self.conf = conf
        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def sources(self, x):
        """ Returns source layers. """
        sources = list(self.resnet(x))
        for block in self.extras:
            sources.append(block(sources[-1]))

        return sources

    def forward(self, x):
        sources = self.sources(x)
        loc = self.loc(sources)
        conf = self.conf(sources)
        loc = [l.permute(0, 2, 3, 1).contiguous() for l in loc]
        conf = [c.permute(0, 2, 3, 1).contiguous() for c in conf]

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
            self.load_state_dict(torch.load(
                base_file, map_location=lambda storage, loc: storage
            ))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def build_ssd(phase='train', size=320, num_classes=21, body='resnet34',
              freeze_batchnorm=False, extra_config='normal'):
    """ Factory function to build SSD. """
    if body == 'resnet34':
        resnet = resnet34_reduced(pretrained=True, atrous=False,
                                  freeze_batchnorm=freeze_batchnorm)
        source_channels = [128, 256, 512]
    elif body == 'resnet50':
        resnet = resnet50_reduced(pretrained=True, atrous=False,
                                  freeze_batchnorm=freeze_batchnorm)
        source_channels = [512, 1024, 2048]
    elif body == 'resnet101':
        resnet = resnet101_reduced(pretrained=True, atrous=False,
                                   freeze_batchnorm=freeze_batchnorm)
        source_channels = [512, 1024, 2048]
    else:
        raise ValueError('Body {} is not supported'.format(body))
    cfg, extras, loc_layers, conf_layers = multibox(size, num_classes,
                                                    source_channels,
                                                    extra_config)
    ssd = SSD(phase, cfg, resnet, extras, loc_layers, conf_layers,
              num_classes, size)

    return ssd


def main():
    size = 320
    #  model = build_ssd(size=size, body='resnet34')
    model = resnet34_reduced(pretrained=True)
    model = model.cuda()
    model.train()
    blob = Variable(torch.randn(16, 3, size, size).cuda())
    output = model(blob)
    #  sources = model.sources(blob)
    #  print([s.size() for s in sources])
    from IPython import embed; embed()


if __name__ == '__main__':
    main()
