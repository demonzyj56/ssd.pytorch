import torch
import torch.nn as nn
from torch.autograd import Function


class DRAELossFunction(Function):
    def __init__(self, lamb=0.1, size_average=True):
        super(DRAELossFunction, self).__init__()
        self.lamb = lamb
        self.size_average = size_average

    def forward(self, input, target):
        buffer = input.new().resize_as_(input).copy_(input)
        self.dims = input.size()
        buffer.add_(-1, target)
        buffer = buffer.view(buffer.size(0), -1)  # reshape the error into 2-D
        self.error = buffer.new().resize_as_(buffer).copy_(buffer)  # raw error
        # print self.error
        torch.mul(buffer, buffer, out=buffer)  # squared error
        Err = torch.sum(buffer, 1)  # reconstruction error of each sample (summer square error)
        del buffer

        # discriminative labelling
        self.sErr, self.idx = torch.sort(Err, 0)
        _, self.reorder_idx = torch.sort(self.idx, 0)
        self.idx.resize_(self.idx.size(0))
        self.reorder_idx.resize_(self.reorder_idx.size(0))
        optObj = 10000.0

        inlierMeanErr = self.sErr[0]
        outlierMeanErr = torch.mean(self.sErr[1:], 0)[0]
        allMeanErr = torch.mean(self.sErr, 0)[0]
        Sb = torch.sum((self.sErr.add(-allMeanErr[0])) ** 2, 0)

        for i in range(Err.size(0) - 1):
            inlierErr = self.sErr[:i + 1]
            # inlierMeanErr = torch.mean(inlierErr, 0)[0]
            outlierErr = self.sErr[i + 1:]
            # outlierMeanErr = torch.mean(outlierErr, 0)[0]
            Sw1 = torch.sum((inlierErr.add(-inlierMeanErr[0])) ** 2, 0)
            Sw2 = torch.sum((outlierErr.add(-outlierMeanErr[0])) ** 2, 0)


            obj = (Sw1 + Sw2) / Sb
            if obj[0, 0] < optObj:
                optObj = obj[0, 0]
                self.T_idx = i + 1
                self.inlierMeanErr = inlierMeanErr
                self.outlierMeanErr = outlierMeanErr
                self.allMeanErr = allMeanErr
                self.Sw1 = Sw1
                self.Sw2 = Sw2
                self.Sb = Sb

            inlierMeanErr = (inlierMeanErr * (i + 1.) + self.sErr[i + 1]) / (i+2.)
            if i != Err.size(0) - 2:
                outlierMeanErr = (outlierMeanErr*(Err.size(0) - i - 1.)-self.sErr[i+1]) / (Err.size(0)-i-2.)
            else:
                pass

        # print self.T_idx, self.inlierMeanErr, self.outlierMeanErr

        output = self.sErr[:self.T_idx].sum()  # only positive label is considered
        if self.size_average:
            output = output / self.T_idx
            # output = output/input.size(1)+self.lamb*optObj
        output = output + self.lamb * optObj

        self.save_for_backward(input, target)
        # print self.error
        return input.new((output,))

    def backward(self, grad_output):
        input, target = self.saved_tensors
        # gradient for reconstruction error part
        grad_input = self.error.new().resize_as_(self.error).copy_(self.error)
        if self.size_average:
            grad_input.mul_(1. / self.T_idx)
        grad_input[self.idx[self.T_idx:]] = 0  # only the gradient of positive sample is backwarded
        # grad_input.mul_(2. / input.size(1))
        grad_input.mul_(2.)

        # print grad_input

        # gradient for the discriminative part
        inlierErr = self.sErr[:self.T_idx]
        outlierErr = self.sErr[self.T_idx:]
        Sw = self.Sw1 + self.Sw2
        Sb2 = self.Sb ** 2
        inlierGrad = inlierErr.add(-self.inlierMeanErr[0]).mul(self.Sb[0, 0]).add(
            -inlierErr.add(-self.allMeanErr[0]).mul(Sw[0, 0])).mul(1. / Sb2[0, 0])
        inlierGrad.mul_(2.0 * 2.0 * self.lamb)  # constant factor
        inlierGrad = inlierGrad.repeat(1, self.error.size(1))
        outlierGrad = outlierErr.add(-self.outlierMeanErr[0]).mul(self.Sb[0, 0]).add(
            -outlierErr.add(-self.allMeanErr[0]).mul(Sw[0, 0])).mul(1. / Sb2[0, 0])
        outlierGrad.mul_(2.0 * 2.0 * self.lamb)
        outlierGrad = outlierGrad.repeat(1, self.error.size(1))
        sGrad = torch.cat([inlierGrad, outlierGrad], 0)
        del inlierGrad, outlierGrad
        hGrad = sGrad[self.reorder_idx]
        del sGrad
        hGrad.mul_(self.error)

        # overall gradient
        grad_input.add_(1., hGrad)
        del hGrad

        # reshape the grad into the original shape of input
        grad_input = grad_input.view(self.dims)

        if grad_output[0] != 1:
            grad_input.mul_(grad_output[0])

        return grad_input, None


class DRAELoss(nn.Module):

    def __init__(self, lamb, size_average=True):
        super(DRAELoss, self).__init__()
        self.lamb = lamb
        self.size_average = size_average

    def forward(self, input, target):
        return DRAELossFunction(self.lamb, self.size_average)(input, target)