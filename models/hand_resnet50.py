import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['resnet50_prune']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, cfg_bef=None):
        super(Bottleneck, self).__init__()
        self.cfg_bef = cfg_bef
        self.cfg = cfg
        if cfg_bef != None:
            self.conv1 = nn.Conv2d(cfg_bef, cfg[1], kernel_size=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[2])
        if downsample != None:
            self.conv3 = nn.Conv2d(cfg[2], cfg[3], kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(cfg[3])
        else:
            if cfg_bef != None:
                self.conv3 = nn.Conv2d(cfg[2], cfg_bef, kernel_size=1, bias=False)
                self.bn3 = nn.BatchNorm2d(cfg_bef)
            else:
                self.conv3 = nn.Conv2d(cfg[2], cfg[0], kernel_size=1, bias=False)
                self.bn3 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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

        out = out.clone() + residual
        out = self.relu(out)

        return out

    def get_output_c(self):
        if self.downsample != None:
            return self.cfg[3]
        elif self.cfg_bef != None:
            return self.cfg_bef
        else:
            return self.cfg[0]


class ResNet(nn.Module):

    def __init__(self, block, layers, cfg, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1, cfg_bef = self._make_layer(block, cfg[0], cfg[1:11], 64, layers[0])
        self.layer2, cfg_bef = self._make_layer(block, cfg_bef, cfg[10:23], 128, layers[1], stride=2)
        self.layer3, cfg_bef = self._make_layer(block, cfg_bef, cfg[22:41], 256, layers[2], stride=2)
        self.layer4, cfg_bef = self._make_layer(block, cfg_bef, cfg[40:50], 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(cfg_bef, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal(m.weight, mode='fan_out')
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, cfg_bef, cfg, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(cfg_bef, cfg[3],
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(cfg[3]),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[:4], stride, downsample, cfg_bef=cfg_bef))
        cfg_bef = block(self.inplanes, planes, cfg[:4], stride, downsample).get_output_c()
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == 1:
                layers.append(block(self.inplanes, planes, cfg[3*i:3*(i+1)+1], cfg_bef=cfg_bef))
                cfg_bef = block(self.inplanes, planes, cfg[3*i:3*(i+1)+1]).get_output_c()
            else:
                layers.append(block(self.inplanes, planes, cfg[3*i:3*(i+1)+1], cfg_bef=cfg_bef))
                cfg_bef = block(self.inplanes, planes, cfg[3*i:3*(i+1)+1], cfg_bef=cfg_bef).get_output_c()

        return nn.Sequential(*layers), cfg_bef

    def forward(self, x):
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


def resnet50_prune(pretrained=False, cfg=None):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if cfg is None:
        cfg_official = [[64], [64, 64, 64], [256, 64, 64] * 2, [256, 128, 128], [512, 128, 128] * 3,
                    [512, 256, 256], [1024, 256, 256] * 5, [1024, 512, 512], [2048, 512, 512] * 2]
        cfg_official = [item for sublist in cfg_official for item in sublist]
        assert len(cfg_official) == 49, "Length of cfg_official is not right"
        cfg = cfg_official
    model = ResNet(Bottleneck, [3, 4, 6, 3], cfg)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model