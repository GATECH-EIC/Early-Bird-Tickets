import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from .quantize import WAGEQuantizer, Q
from .wage_initializer import wage_init_

__all__ = ['vgg_lp']

defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class vgg_lp(nn.Module):
    def __init__(self, bits_A, bits_E, bits_W, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(vgg_lp, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self.bits_W = bits_W
        self.quant = WAGEQuantizer(bits_A, bits_E)

        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-1], num_classes),
            WAGEQuantizer(-1, bits_E, "bf-loss") # only quantize backward pass
        )
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                layers += [self.quant]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data = Q(m.weight.data, self.bits_W)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        self.weight_scale = {}
        for name, param in self.named_parameters():
            self.weight_scale[name] = 1

    def _wage_initialize_weights(self):
        self.weight_scale = {}
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.data.shape) > 3:
                wage_init_(param, self.bits_W, name, self.weight_scale, factor=1.0)
                param.data = Q(param.data, self.bits_W)
            self.weight_scale[name] = 1

if __name__ == '__main__':
    net = vgg_lp(8, 8, 3)
    x = Variable(torch.FloatTensor(16, 3, 40, 40))
    y = net(x)
    print(y.data.shape)