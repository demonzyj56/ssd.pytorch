""" Autoencoder for reconstruction. """
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from numpy import argmin
from ..box_utils import match
from ..functions import PriorBox

mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
}


def prepare_positive_targets(priors, targets, batch_size, ov_thresh, cfg):
    """ Given priors and targets for an image, do the following:
    Find the location of each positive targets, then unmap into separate
    masks for each feature map.  Each feature map is of size
    [batch_size, mbox_size, height, width].
    """
    conf_t= torch.LongTensor(batch_size, priors.size(0)).cuda()
    loc_t = torch.Tensor(batch_size, priors.size(0), 4).cuda()
    for idx in range(batch_size):
        truths = targets[idx][:, :-1].data
        labels = targets[idx][:, -1].data
        defaults = priors.data
        match(ov_thresh, truths, defaults, cfg['variance'], labels,
              loc_t, conf_t, idx)
    conf_t = conf_t.gt(0)

    feature_segs = zip(cfg['feature_maps'], mbox[str(cfg['min_dim'])])
    feature_size = [i ** 2 * j for (i, j) in feature_segs]
    assert sum(feature_size) == conf_t.size(1), \
        'Total feature size is {} while conf_t gives {}'.format(sum(feature_size), conf_t.size(1))
    gt_masks = []
    accum = 0
    for (i, j) in zip(cfg['feature_maps'], mbox[str(cfg['min_dim'])]):
        gt_masks.append(
                conf_t[:, accum:(accum+i**2*j)].contiguous().view(-1, j, i, i)
            )
        accum += i ** 2 * j
    return gt_masks


class ConvolutionalAutoencoder(nn.Module):

    def __init__(self, in_channels):
        super(ConvolutionalAutoencoder, self).__init__()
        self.encode = nn.Conv2d(in_channels, in_channels//8, kernel_size=1,
                                stride=1, padding=0)
        self.decode = nn.Conv2d(in_channels//8, in_channels, kernel_size=1,
                                stride=1, padding=0)

    def forward(self, xx):
        # x = F.dropout(self.encode(xx), training=self.training)
        x = self.encode(xx)
        x = F.relu(self.decode(x))

        return torch.pow(xx - x, 2).sum(dim=1)


class MnistAutoencoder(nn.Module):

    def __init__(self, in_channels, noise_prob=-1):
        super(MnistAutoencoder, self).__init__()
        self.encode1 = self.conv1x1(in_channels, in_channels*2)
        self.encode2 = self.conv1x1(in_channels*2, in_channels//2)
        self.encode3 = self.conv1x1(in_channels//2, in_channels//8)
        self.decode3 = self.conv1x1(in_channels//8, in_channels//2)
        self.decode2 = self.conv1x1(in_channels//2, in_channels*2)
        self.decode1 = self.conv1x1(in_channels*2, in_channels)
        # self.encode = self.conv1x1(in_channels, in_channels//8)
        # self.decode = self.conv1x1(in_channels//8, in_channels)
        self.noise_prob = noise_prob

    def forward(self, xx):
        if self.noise_prob > 0:
            x = F.dropout(xx, p=self.noise_prob, training=self.training)
        else:
            x = xx
        # x = self.decode(self.encode(x))
        x = F.relu(self.encode1(x), inplace=True)
        x = F.relu(self.encode2(x), inplace=True)
        x = self.encode3(x)
        x = F.relu(self.decode3(x), inplace=True)
        x = F.relu(self.decode2(x), inplace=True)
        x = F.relu(self.decode1(x), inplace=True)

        return torch.pow(xx-x, 2).sum(dim=1)

    def conv1x1(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=1, padding=0)


class DiscriminativeReconstructionLoss(nn.Module):

    def __init__(self, lamb, cfg, ov_thresh):
        super(DiscriminativeReconstructionLoss, self).__init__()
        self.cfg = cfg
        self.ov_thresh = ov_thresh
        self.lamb = lamb
        _priorbox = PriorBox(cfg)
        self.priors = Variable(_priorbox.forward(), volatile=True)

    def forward(self, x, targets):
        gt_masks = prepare_positive_targets(
            self.priors, targets, x[0].size(0), self.ov_thresh, self.cfg
        )
        gt_masks = [Variable(mask.sum(dim=1).gt(0), requires_grad=False) for mask in gt_masks]
        assert len(x) == len(gt_masks), \
            'Length of input is {} while num of masks is {}'.format(len(x), len(gt_masks))
        assert all([xx.size() == mask.size() for (xx, mask) in zip(x, gt_masks)])
        losses = []
        num_samples = []
        samples_selected = []
        for (xx, mask) in zip(x, gt_masks):
            pos = torch.masked_select(xx, mask)
            num_samples.append(pos.numel())
            if pos.numel() == 0:
                losses.append(Variable(torch.zeros(1), requires_grad=False).cuda())
                samples_selected.append(0)
            elif pos.numel() <= 2:
                losses.append(pos.mean())
                samples_selected.append(pos.numel())
            else:
                reg, idx = self.regularity_term(pos)
                recon_loss = pos[idx].mean() + self.lamb * reg
                losses.append(recon_loss)
                samples_selected.append(idx.numel())
        # print(num_samples)
        # print(samples_selected)
        return losses

    def regularity_term(self, x):
        assert len(x) > 2
        regularities = []
        x, idx = torch.sort(x, descending=False)
        for i in range(1, x.numel()):
            pos_variance = torch.var(x[:i], unbiased=False)*i if i > 1 else 0.
            neg_variance = torch.var(x[i:], unbiased=False)*(x.numel()-i) if i < x.numel()-1 else 0.
            regularities.append(pos_variance + neg_variance)
        select = argmin([r.data[0] for r in regularities])
        pos_idx = idx[:(select+1)]

        return regularities[select] / x.var(unbiased=False) / x.numel(), pos_idx

    def samples(self, feature_maps, targets):
        gt_masks = prepare_positive_targets(
            self.priors, targets, feature_maps[0].size(0), self.ov_thresh, self.cfg
        )
        gt_masks = [mask.sum(dim=1).gt(0) for mask in gt_masks]
        samples = []
        for feature_map, mask in zip(feature_maps, gt_masks):
            sample = []
            for num in range(mask.size(0)):
                for h in range(mask.size(1)):
                    for w in range(mask.size(2)):
                        if mask[num, h, w]:
                            sample.append(feature_map.data[num, :, h, w].contiguous().view(1, -1))
            if len(sample) > 0:
                samples.append(torch.cat(sample, 0).cpu().numpy())
            else:
                samples.append([])

        return samples


class SSDReconstruction(nn.Module):

    def __init__(self, ssd, p=-1):
        super(SSDReconstruction, self).__init__()
        self.ssd = ssd
        for param in self.ssd.parameters():
            param.requires_grad = False
        self.ae = nn.ModuleList([
            MnistAutoencoder(c.in_channels, noise_prob=p) for c in ssd.conf
        ])

    def forward(self, x):
        feature_maps = self.ssd_feature_maps(x)
        return [a(f) for (f, a) in zip(feature_maps, self.ae)]

    def ssd_feature_maps(self, x):
        # TODO(leoyolo): subject to change
        sources = list()
        ssd = self.ssd
        for k in range(23):
            x = ssd.vgg[k](x)
        s = ssd.L2Norm(x)
        sources.append(s)
        for k in range(23, len(ssd.vgg)):
            x = ssd.vgg[k](x)
        sources.append(x)
        for k, v in enumerate(ssd.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        return sources
