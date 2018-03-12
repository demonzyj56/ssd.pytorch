""" Non-local block used in Non-Local Neural Network. """
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlock(nn.Module):

    def __init__(self, in_channels, subsample=False):
        super(NonLocalBlock, self).__init__()
        self.mid_channels = in_channels // 2
        self.subsample = subsample
        self.conv1 = nn.Conv3d(in_channels, self.mid_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels, self.mid_channels, kernel_size=1)
        self.conv3 = nn.Conv3d(in_channels, self.mid_channels, kernel_size=1)
        self.conv4 = nn.Conv1d(self.mid_channels, in_channels, kernel_size=1)

        self.reset_parameters()

    def forward(self, x):
        """ The input tensor is a 5-dim Variable with configuration
        [batch_size, channels, depth, height, width].
        Output non-local transformation with the same size. """
        embed1 = self.conv1(x).view(x.size(0), self.mid_channels, -1).permute(0, 2, 1)
        embed2 = self.conv2(x)
        embed3 = self.conv3(x)
        if self.subsample:
            embed2 = F.max_pool3d(embed2, kernel_size=(1, 2, 2))
            embed3 = F.max_pool3d(embed3, kernel_size=(1, 2, 2))
        embed2 = embed2.view(x.size(0), self.mid_channels, -1)
        embed3 = embed3.view(x.size(0), self.mid_channels, -1).permute(0, 2, 1)
        kernel = embed1.matmul(embed2)
        kernel = F.softmax(kernel, dim=2)
        kernel = kernel.matmul(embed3).permute(0, 2, 1)
        kernel = self.conv4(kernel)
        x = x + kernel.view(x.size())

        return x

    def reset_parameters(self):
        def init(m):
            nn.init.kaiming_normal(m.weight.data)
            m.bias.data.zero_()
        init(self.conv1)
        init(self.conv2)
        init(self.conv3)
        init(self.conv4)


if __name__ == '__main__':
    from torch.autograd import Variable
    from IPython import embed
    batch_size = 16
    channels = 256
    depth = 2
    height = 64
    width = 64
    nl = NonLocalBlock(channels, subsample=True)
    nl.cuda()
    block = Variable(torch.randn(batch_size, channels, depth, height, width).cuda())
    output = nl(block)
    embed()
