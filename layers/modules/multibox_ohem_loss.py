import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import match, log_sum_exp

class MultiBoxOHEMLoss(nn.Module):
    """SSD Weighted OHEM Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
            size_average: whether to average the output loss by N.  If not,
            then return N at the last position.
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, num_samples, overlap_thresh, variance,
                 use_gpu=True, size_average=True):
        super(MultiBoxOHEMLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.threshold = overlap_thresh
        self.size_average = size_average
        self.variance = variance

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)

        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Online Hard Example Mining
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        #  selected = idx_rank.lt(self.num_samples).unsqueeze(2).expand_as(conf_data)
        selected = idx_rank.lt(self.num_samples)
        conf_p = conf_data[selected.unsqueeze(2).expand_as(conf_data)].view(-1, self.num_classes)
        targets_weighted = conf_t[selected]
        loss_c_ohem = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        N = pos.data.sum()

        if self.size_average:
            return loss_l/N, loss_c_ohem/(self.num_samples*num)
        else:
            return loss_l, loss_c_ohem, N

