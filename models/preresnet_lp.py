from __future__ import absolute_import
import math

import torch.nn as nn
from .quantize import WAGEQuantizer, Q
from .wage_initializer import wage_init_
from .channel_selection import channel_selection


__all__ = ['resnet_lp']

"""
preactivation resnet with bottleneck design.
"""

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, bits_A, bits_E, inplanes, planes, cfg, stride=1, downsample=None):
        """
        Initialize the batch.

        Args:
            self: (todo): write your description
            bits_A: (todo): write your description
            bits_E: (int): write your description
            inplanes: (todo): write your description
            planes: (todo): write your description
            cfg: (todo): write your description
            stride: (int): write your description
            downsample: (todo): write your description
        """
        super(Bottleneck, self).__init__()
        self.quant = WAGEQuantizer(bits_A, bits_E)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Perform forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class resnet_lp(nn.Module):
    def __init__(self, bits_A, bits_E, bits_W, depth=164, dataset='cifar10', cfg=None):
        """
        Initialize layer layers.

        Args:
            self: (todo): write your description
            bits_A: (todo): write your description
            bits_E: (int): write your description
            bits_W: (int): write your description
            depth: (float): write your description
            dataset: (todo): write your description
            cfg: (todo): write your description
        """
        super(resnet_lp, self).__init__()
        self.bits_A = bits_A
        self.bits_E = bits_E
        self.bits_W = bits_W
        self.quant = WAGEQuantizer(bits_A, bits_E)

        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        n = (depth - 2) // 9
        block = Bottleneck

        if cfg is None:
            # Construct config variable.
            cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n, cfg = cfg[0:3*n])
        self.layer2 = self._make_layer(block, 32, n, cfg = cfg[3*n:6*n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg = cfg[6*n:9*n], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.select = channel_selection(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        if dataset == 'cifar10':
            self.fc = nn.Sequential(
                nn.Linear(cfg[-1], 10),
                WAGEQuantizer(-1, self.bits_E, "bf-loss") # only quantize backward pass
            )
        elif dataset == 'cifar100':
            self.fc = nn.Sequential(
                nn.Linear(cfg[-1], 100),
                WAGEQuantizer(-1, self.bits_E, "bf-loss")
            )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights.

        Args:
            self: (todo): write your description
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data = Q(m.weight.data, self.bits_W)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
        self.weight_scale = {}
        for name, param in self.named_parameters():
            self.weight_scale[name] = 1

    def _wage_initialize_weights(self):
        """
        Initialize the weights.

        Args:
            self: (todo): write your description
        """
        self.weight_scale = {}
        for name, param in self.named_parameters():
            if 'weight' in name and not 'bn' in name and not 'downsample' in name:
                wage_init_(param, self.bits_W, name, self.weight_scale, factor=1.0)
                param.data = Q(param.data, self.bits_W)
            if 'bn' in name:
                if 'weight' in name: param.data.fill_(0.5); self.weight_scale[name] = 1
                if 'bias' in name: param.data.zero_(); self.weight_scale[name] = 1
            if 'downsample' in name:
                self.weight_scale[name] = 1
            if 'bias' in name:
                param.data.zero_()
                self.weight_scale[name] = 1

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        """
        Make a layer.

        Args:
            self: (todo): write your description
            block: (todo): write your description
            planes: (todo): write your description
            blocks: (todo): write your description
            cfg: (todo): write your description
            stride: (int): write your description
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.bits_A, self.bits_E, self.inplanes, planes, cfg[0:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.bits_A, self.bits_E, self.inplanes, planes, cfg[3*i: 3*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward computation of the layer

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.quant(x)
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)
        x = self.quant(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    net = resnet_lp(8, 8, 8, 20)
    import torch
    for i, m in enumerate(net.modules()):
        if isinstance(m, nn.Conv2d):
            print(m.weight.data.numel())
            mask = m.weight.data.abs().gt(0).float()
            print(torch.sum(mask))
            print(m.weight.data.view(-1).abs().clone())
            break