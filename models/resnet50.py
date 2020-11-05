import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .channel_selection import channel_selection

__all__ = ['resnet50_official']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            inplanes: (todo): write your description
            planes: (todo): write your description
            cfg: (todo): write your description
            stride: (int): write your description
            downsample: (todo): write your description
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, cfg, num_classes=1000):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            block: (todo): write your description
            layers: (list): write your description
            cfg: (todo): write your description
            num_classes: (int): write your description
        """
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, cfg[1:10], 64, layers[0])
        self.layer2 = self._make_layer(block, cfg[10:22], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[22:40], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, cfg[40:49], 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, cfg, planes, blocks, stride=1):
        """
        Make a layer.

        Args:
            self: (todo): write your description
            block: (todo): write your description
            cfg: (todo): write your description
            planes: (todo): write your description
            blocks: (todo): write your description
            stride: (int): write your description
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3*i:3*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Perform forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

cfg_official = [[64, 64, 64], [256, 64, 64] * 2, [256, 128, 128], [512, 128, 128] * 3, 
                [512, 256, 256], [1024, 256, 256] * 5, [1024, 512, 512], [2048, 512, 512] * 2]
cfg_official = [item for sublist in cfg_official for item in sublist]
assert len(cfg_official) == 48, "Length of cfg_official is not right"


def resnet50_official(depth=50, dataset='ImageNet', cfg=None, pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if cfg == None:
        cfg_official = [[64], [64, 64, 64], [256, 64, 64] * 2, [256, 128, 128], [512, 128, 128] * 3, 
                    [512, 256, 256], [1024, 256, 256] * 5, [1024, 512, 512], [2048, 512, 512] * 2]
        cfg_official = [item for sublist in cfg_official for item in sublist]
        assert len(cfg_official) == 49, "Length of cfg_official is not right"
        cfg = cfg_official
    model = ResNet(Bottleneck, [3, 4, 6, 3], cfg)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model