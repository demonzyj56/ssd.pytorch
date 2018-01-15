""" Autoencoder for reconstruction. """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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


class MyAutoencoder(nn.Module):

    def __init__(self, in_channels, p=-1):
        super(MyAutoencoder, self).__init__()
        self.encode1 = nn.Linear(in_channels, in_channels*2)
        self.encode2 = nn.Linear(in_channels*2, in_channels//2)
        self.encode3 = nn.Linear(in_channels//2, in_channels//8)
        self.decode3 = nn.Linear(in_channels//8, in_channels//2)
        self.decode2 = nn.Linear(in_channels//2, in_channels*2)
        self.decode1 = nn.Linear(in_channels*2, in_channels)
        self.p = p

    def forward(self, xx):
        if self.p > 0:
            x = F.dropout(xx, p=self.p, training=self.training)
        else:
            x = xx
        x = F.sigmoid(self.encode1(x))
        x = F.sigmoid(self.encode2(x))
        x = self.encode3(x)
        x = F.sigmoid(self.decode3(x))
        x = F.sigmoid(self.decode2(x))
        x = F.relu(self.decode1(x))

        return x


class SSDReconstruction(nn.Module):

    def __init__(self, ssd, cfg, ov_thresh, accum=-1, p=-1):
        super(SSDReconstruction, self).__init__()
        self.ssd = ssd
        for param in self.ssd.parameters():
            param.requires_grad = False
        self.ae = nn.ModuleList([
            MyAutoencoder(c.in_channels, p=p) for c in ssd.conf
        ])
        self.cfg = cfg
        self.ov_thresh = ov_thresh
        self.accum = accum
        self.priors = Variable(PriorBox(cfg).forward(), volatile=True)
        self.in_channels = [c.in_channels for c in ssd.conf]
        if accum > 0:
            self.accumulated_features = [self._empty_tensor() for _ in self.in_channels]
        else:
            self.accumulated_features = None

    def forward(self, x, targets):
        feature_maps = self.ssd_feature_maps(x)
        features = self.extract_features(feature_maps, targets)
        if self.accum <= 0:
            # compute loss immediately
            features = [Variable(f).cuda() for f in features]
            return_pairs = []
            for (f, a) in zip(features, self.ae):
                if self._valid_feature(f):
                    return_pairs.append((f, a(f)))
                else:
                    return_pairs.append((None, None))
        else:
            # postpone until enough samples are accumulated
            for idx in range(len(features)):
                if self._valid_feature(self.accumulated_features[idx]):
                    if self._valid_feature(features[idx]):
                        self.accumulated_features[idx] = torch.cat((self.accumulated_features[idx], features[idx]), dim=0)
                else:
                    if self._valid_feature(features[idx]):
                        self.accumulated_features[idx] = features[idx]
            return_pairs = []
            for idx, channels in enumerate(self.in_channels):
                if self._valid_feature(self.accumulated_features[idx]) and self.accumulated_features[idx].size(0) >= self.accum:
                    feature = Variable(self.accumulated_features[idx]).cuda()
                    return_pairs.append((feature, self.ae[idx](feature)))
                    self.accumulated_features[idx] = self._empty_tensor()
                else:
                    return_pairs.append((None, None))

        return return_pairs

    def forward_full_batch(self, x, targets):
        feature_maps = self.ssd_feature_maps(x)
        features = self.extract_features(feature_maps, targets)
        return_pairs = []
        if True:
            # compute loss immediately
            features = [Variable(f).cuda() for f in features]
            return_pairs = []
            for (f, a) in zip(features, self.ae):
                if self._valid_feature(f):
                    return_pairs.append((f, a(f)))
                else:
                    return_pairs.append((None, None))

        return return_pairs

    def show_hand(self):
        return_pairs = []
        for feature, ae in zip(self.accumulated_features, self.ae):
            if self._valid_feature(feature):
                f = Variable(feature).cuda()
                return_pairs.append((f, ae(f)))
            else:
                return_pairs.append((None, None))

        self.accumulated_features = None
        return return_pairs

    def _valid_feature(self, f):
        return len(f) > 0

    def _empty_tensor(self):
        return torch.zeros(0)

    def ssd_feature_maps(self, x):
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

    def extract_features(self, feature_maps, targets):
        gt_masks = prepare_positive_targets(
            self.priors, targets, feature_maps[0].size(0), self.ov_thresh, self.cfg
        )
        gt_masks = [mask.sum(dim=1).gt(0) for mask in gt_masks]
        samples = []
        for feature_map, mask in zip(feature_maps, gt_masks):
            feature_map = feature_map.data.permute(0, 2, 3, 1)
            mask = mask.unsqueeze(-1).expand_as(feature_map)
            sample = torch.masked_select(feature_map, mask).contiguous().view(-1, feature_map.size(-1))
            samples.append(sample)

        return samples


