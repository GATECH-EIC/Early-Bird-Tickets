import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d
from torch.nn.grad import conv2d_input, conv2d_weight
from torch.nn.modules.utils import _pair
from torch.autograd import Function

from .quantize_2 import calculate_qparams, quantize, quantize_grad, Quantize

class WeightQuantFunc(Function):
    @staticmethod
    def forward(self, weight, num_bits_weight):
        with torch.no_grad():
            # q_weight
            if num_bits_weight is not None and num_bits_weight < 32:
                q_weight = quantize(
                weight, num_bits=num_bits_weight, dequantize=True, flatten_dims=(1,-1), reduce_dim=None, signed=True)
            else:
                q_weight = weight
        return q_weight

    @staticmethod
    def backward(self, grad_output):
        with torch.no_grad():
            grad_weight = grad_output
        return grad_weight, None

def quant_weight(weight, num_bits_weight=8):
    return WeightQuantFunc.apply(weight, num_bits_weight)

def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_bits_grad=None):
    out1 = F.conv2d(input.detach(), weight, bias,
                    stride, padding, dilation, groups)
    out2 = F.conv2d(input, weight.detach(), bias.detach() if bias is not None else None,
                    stride, padding, dilation, groups)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()

class new_conv(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, num_bits=8, num_bits_weight=8, num_bits_grad=8, input_signed=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(new_conv, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias=False)

        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight
        self.num_bits_grad = num_bits_grad
        self.input_signed = input_signed

        self.quant_input = Quantize(num_bits=self.num_bits, shape_measure=(1,1,1,1,), flatten_dims=(1, -1), dequantize=True, input_signed=self.input_signed, stochastic=False, momentum=0.1)

    def forward(self, input):
        q_input = self.quant_input(input)
        q_weight = quant_weight(self.weight, num_bits_weight=self.num_bits_weight)
        q_bias = None
        if self.num_bits_grad is None:
            q_output = F.conv2d(q_input, q_weight, bias=q_bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            q_output = conv2d_biprec(q_input, q_weight, bias=q_bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, num_bits_grad=self.num_bits_grad)
        return q_output