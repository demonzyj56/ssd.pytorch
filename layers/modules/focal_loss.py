""" Focal Loss for Dense Object Detection.
This is a PyTorch implementation of focal_loss layer. """
import torch
import torch.nn as nn
from torch.autograd import Variable, Function


class FocalLossFunction(Function):

    @staticmethod
    def forward(ctx, prediction, target, gamma, alpha):
        """
        FL = -(1-p_c)^\gamma * log p_c,
            where c is the target class.
        """
        ctx.gamma = gamma
        ctx.alpha = alpha
        ctx.log_p = prediction.clone()
        ctx.weight = prediction.new(prediction.size(0)).fill_(1-alpha)
        ctx.weight.masked_fill_(target.gt(0), alpha)
        buf = prediction.clone()
        lse = prediction.new(prediction.size(0))  # log-sum-exp
        loss = prediction.new(1)
        ctx.save_for_backward(target)

        x_max, _ = buf.max(dim=1, keepdim=True)
        buf.sub_(x_max).exp_()
        torch.sum(buf, 1, out=lse)
        lse.log_().add_(x_max)
        ctx.log_p.sub_(lse.view(-1, 1))
        torch.gather(ctx.log_p, 1, target, out=lse)
        lse.squeeze_()
        ctx.log_pt = lse.clone()
        lse.exp_().neg_().add_(1).pow_(gamma).neg_().mul_(ctx.weight)
        loss.fill_(torch.dot(lse, ctx.log_pt))

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        target, = ctx.saved_tensors
        pt = ctx.log_pt.clone()
        pt.exp_()
        buf = pt.clone()
        buf.mul_(ctx.log_pt).mul_(ctx.gamma).add_(pt).sub_(1)
        pt.neg_().add_(1).pow_(ctx.gamma-1).mul_(buf).mul_(ctx.weight)
        buf.fill_(1)

        grad = ctx.log_p.clone()
        grad.exp_().neg_().scatter_add_(
            1, target, buf.view(-1, 1)).mul_(pt.view(-1, 1))
        grad_variable = Variable(grad)

        return grad_variable * grad_output, None, None, None


class FocalLoss(nn.Module):
    """ Wrap around FocalLossFunction as a module.
    Assume alpha is a constant for balancing focal loss. """

    def __init__(self, gamma, alpha, ignore_label=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_label = ignore_label

    def forward(self, prediction, target):
        """ Assumes that prediction is a two-dim blob where the first dimension
        contains all samples. """
        selected = target.ne(self.ignore_label)
        prediction = prediction[
            selected.expand_as(prediction)].view(-1, prediction.size(1))
        target = target[selected].view(-1, 1)
        loss = FocalLossFunction.apply(prediction, target,
                                       self.gamma, self.alpha)

        return loss


if __name__ == '__main__':
    # For debug purpose
    import numpy as np
    from torch.autograd import gradcheck
    # from layers.modules.multibox_focal_loss import FocalLoss as FocalLossOrig
    torch.set_default_tensor_type('torch.DoubleTensor')
    r = np.random.randint(0, high=21, size=1000)
    target = Variable(torch.LongTensor(r).view(-1, 1), requires_grad=False)
    prediction = Variable(torch.randn(1000, 21), requires_grad=True)
    gamma = 5*torch.rand(1)[0]+1
    alpha = torch.rand(1)[0]
    #  fl_orig = FocalLossOrig(gamma, alpha)
    fl = FocalLoss(gamma, alpha)
    #  orig = fl_orig(prediction, target)
    this = fl(prediction, target)
    #  print((orig-this).abs().data[0])
    test = gradcheck(FocalLossFunction.apply, (prediction, target, gamma, alpha),
                     eps=1e-6, atol=1e-9)
    print(test)
