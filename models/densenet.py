import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .channel_selection import channel_selection


__all__ = ['densenet']

"""
densenet with basic block.
"""

class BasicBlock(nn.Module):
    def __init__(self, inplanes, cfg, expansion=1, growthRate=12, dropRate=0):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            inplanes: (todo): write your description
            cfg: (todo): write your description
            expansion: (todo): write your description
            growthRate: (todo): write your description
            dropRate: (todo): write your description
        """
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out

class Transition(nn.Module):
    def __init__(self, inplanes, outplanes, cfg):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            inplanes: (todo): write your description
            outplanes: (int): write your description
            cfg: (todo): write your description
        """
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Evaluate forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out

class densenet(nn.Module):

    def __init__(self, depth=40, 
        dropRate=0, dataset='cifar10', growthRate=12, compressionRate=1, cfg = None):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            depth: (float): write your description
            dropRate: (todo): write your description
            dataset: (todo): write your description
            growthRate: (todo): write your description
            compressionRate: (str): write your description
            cfg: (todo): write your description
        """
        super(densenet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3
        block = BasicBlock

        self.growthRate = growthRate
        self.dropRate = dropRate

        if cfg == None:
            cfg = []
            start = growthRate*2
            for i in range(3):
                cfg.append([start+12*i for i in range(n+1)])
                start += growthRate*12
            cfg = [item for sub_list in cfg for item in sub_list]

        assert len(cfg) == 3*n+3, 'length of config variable cfg should be 3n+3'

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_denseblock(block, n, cfg[0:n])
        self.trans1 = self._make_transition(compressionRate, cfg[n])
        self.dense2 = self._make_denseblock(block, n, cfg[n+1:2*n+1])
        self.trans2 = self._make_transition(compressionRate, cfg[2*n+1])
        self.dense3 = self._make_denseblock(block, n, cfg[2*n+2:3*n+2])
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.select = channel_selection(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        if dataset == 'cifar10':
            self.fc = nn.Linear(cfg[-1], 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(cfg[-1], 100)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, cfg):
        """
        Make a dense block.

        Args:
            self: (todo): write your description
            block: (todo): write your description
            blocks: (todo): write your description
            cfg: (todo): write your description
        """
        layers = []
        assert blocks == len(cfg), 'Length of the cfg parameter is not right.'
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, cfg = cfg[i], growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate, cfg):
        """
        Make transition matrix.

        Args:
            self: (todo): write your description
            compressionRate: (todo): write your description
            cfg: (todo): write your description
        """
        # cfg is a number in this case.
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes, cfg)

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.conv1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x