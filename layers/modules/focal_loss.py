""" Focal Loss for Dense Object Detection.
This is a PyTorch implementation of focal_loss layer. """
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma, alpha):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        """ Assumes that input is a two-dim blob where the first dimension
        contains all samples. """
        log_pt = F.log_softmax(input, dim=1)
        log_pt = log_pt.gather(1, target).view(-1)
        pt = log_pt.exp()
        loss = -self.alpha * (1-pt)**self.gamma * log_pt

        return loss.sum()
