import os
import math
import torch
import torch.nn as nn
import torchvision.models
from .channel_selection import channel_selection

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# you need to download the models to ~/.torch/models
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }
models_dir = os.path.expanduser('~/.torch/models')
model_name = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        """
        Initialize batch

        Args:
            self: (todo): write your description
            inplanes: (todo): write your description
            planes: (todo): write your description
            cfg: (todo): write your description
            stride: (int): write your description
            downsample: (todo): write your description
        """
        super(BasicBlock, self).__init__()
        self.select = channel_selection(inplanes)
        self.conv1 = conv3x3(cfg[0], cfg[1], stride)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg[1], planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Perform forward

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        residual = x

        out = self.select(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Initialize the convolutional layer.

        Args:
            self: (todo): write your description
            inplanes: (todo): write your description
            planes: (todo): write your description
            stride: (int): write your description
            downsample: (todo): write your description
        """
        super(Bottleneck, self).__init__()
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward computation. forward.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        residual = x

        out = self.select(x)
        out = self.conv1(out)
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

    def __init__(self, block, layers, num_classes=1000, cfg=None):
        """
        Initialize layer layers.

        Args:
            self: (todo): write your description
            block: (todo): write your description
            layers: (list): write your description
            num_classes: (int): write your description
            cfg: (todo): write your description
        """
        super(ResNet, self).__init__()
        self.inplanes = 64

        if cfg is None:
            # Construct config variable (basic block)
            cfg = [[64], [64, 64]*layers[0], [128, 128]*layers[1], [256, 256]*layers[2], [512, 512]* layers[3]]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], cfg=cfg[0:2*layers[0]])
        self.layer2 = self._make_layer(block, 128, layers[1], cfg=cfg[2*layers[0]: 2*(layers[0]+layers[1])], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cfg=cfg[2*(layers[0]+layers[1]): 2*(layers[0]+layers[1]+layers[2])], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cfg=cfg[2*(layers[0]+layers[1]+layers[2]): 2*(layers[0]+layers[1]+layers[2]+layers[3])], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.select = channel_selection(64 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:2], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[2*i: 2*(i+1)]))

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
        x = self.select(x)
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


def resnet18(pretrained=False, cfg=None, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], cfg=cfg, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet18'])))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet34'])))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet50'])))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet101'])))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet152'])))
    return model