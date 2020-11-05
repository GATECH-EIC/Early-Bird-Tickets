# Code from https://github.com/simochen/model-tools.
import numpy as np

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable


def print_model_param_nums(model=None):
    """
    Prints the number of - thumbnails of the model.

    Args:
        model: (todo): write your description
    """
    if model == None:
        model = torchvision.models.alexnet()
    total = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    return total

def print_model_param_flops(model=None, input_res=224, multiply_adds=True):
    """
    Prints the model for all the model.

    Args:
        model: (todo): write your description
        input_res: (todo): write your description
        multiply_adds: (bool): write your description
    """

    prods = {}
    def save_hook(name):
        """
        Save the hook to a hook hook.

        Args:
            name: (str): write your description
        """
        def hook_per(self, input, output):
            """
            Evaluate the given layer

            Args:
                self: (todo): write your description
                input: (array): write your description
                output: (todo): write your description
            """
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        """
        Simple hook to the model.

        Args:
            self: (todo): write your description
            input: (array): write your description
            output: (todo): write your description
        """
        list_1.append(np.prod(input[0].shape))

    list_2={}
    def simple_hook2(self, input, output):
        """
        Convert a simple hook.

        Args:
            self: (todo): write your description
            input: (array): write your description
            output: (todo): write your description
        """
        list_2['names'] = np.prod(input[0].shape)

    list_conv=[]
    def conv_hook(self, input, output):
        """
        Conv_hook.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            output: (todo): write your description
        """
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        """
        Add a single linear hook.

        Args:
            self: (todo): write your description
            input: (array): write your description
            output: (todo): write your description
        """
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        """
        Emit a hook.

        Args:
            self: (todo): write your description
            input: (str): write your description
            output: (todo): write your description
        """
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        """
        Relu the hook.

        Args:
            self: (todo): write your description
            input: (array): write your description
            output: (todo): write your description
        """
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        """
        Applies a layer.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            output: (todo): write your description
        """
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        """
        Upsample hook.

        Args:
            self: (todo): write your description
            input: (array): write your description
            output: (todo): write your description
        """
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        """
        Initialize the network.

        Args:
            net: (todo): write your description
        """
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    if model == None:
        model = torchvision.models.alexnet()
    foo(model)
    input = Variable(torch.rand(3,input_res,input_res).unsqueeze(0), requires_grad = True)
    out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))

    print('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))

    return total_flops