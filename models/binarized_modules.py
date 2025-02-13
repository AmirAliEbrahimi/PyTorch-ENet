import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np


def Binarize(tensor, quant_mode='det'):
    if quant_mode == 'det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(
            torch.rand(tensor.size()).add(-0.5)).clamp_(
                0, 1).round().mul_(2).add_(-1)


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.margin = 1.0

    def hinge_loss(self, input, target):
        #import pdb; pdb.set_trace()
        output = self.margin - input.mul(target)
        output[output.le(0)] = 0
        return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input, target)


class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction, self).__init__()
        self.margin = 1.0

    def forward(self, input, target):
        output = self.margin - input.mul(target)
        output[output.le(0)] = 0
        self.save_for_backward(input, target)
        loss = output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self, grad_output):
        input, target = self.saved_tensors
        output = self.margin - input.mul(target)
        output[output.le(0)] = 0
        import pdb
        pdb.set_trace()
        grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
        grad_output.mul_(output.ne(0).float())
        grad_output.div_(input.numel())
        return grad_output, grad_output


def Quantize(tensor, quant_mode='det', params=None, numBits=8):
    tensor.clamp_(-2**(numBits - 1), 2**(numBits - 1))
    if quant_mode == 'det':
        tensor = tensor.mul(2**(numBits - 1)).round().div(2**(numBits - 1))
    else:
        tensor = tensor.mul(2**(numBits - 1)).round().add(
            torch.rand(tensor.size()).add(-0.5)).div(2**(numBits - 1))
        quant_fixed(tensor, params)
    return tensor


class BinarizeLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class InputScale(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        scale_min = -1
        scale_max = 1
        input = input.type(torch.FloatTensor).to(input.device)
        Ztmp = scale_min + (scale_max - scale_min) * input / 255
        Zmod = torch.remainder(Ztmp, 0.0078125)
        Xvalue = Ztmp - Zmod
        torch.where(Xvalue == 1, Xvalue - 0.0078125, Xvalue)
        out = Xvalue.type(torch.FloatTensor).to(input.device)
        return out

class SignumActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        Z=torch.sign(input);
        z_i = Z==0
        #torch.where(Z == 0., 1., Z)
        Z[z_i] = 1
        self.signumInput=input
        return Z

    def backward(self,grad_output):
        input, = self.signumInput
        return 2.5*((1/torch.cosh(input))**2)*(grad_output)

class BinarizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


class BinarizeTransposedConv2d(nn.ConvTranspose2d):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeTransposedConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)

        out = nn.functional.conv_transpose2d(input, self.weight, None,
                                             self.stride, self.padding,
                                             self.output_padding,self.groups, self.dilation)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
