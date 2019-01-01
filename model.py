"""
The file containing the brain of the AI.

The model is an Asynchronous Advantage Actor-Critic (A3C) algorithm for the
game Breakout.

Coded by: Sultan Sidhu
January 1, 2019
"""
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as func

# initializing and setting the variance of the tensor of weights


def normalized_columns_initializer(weights, std=1.0):
    """Returns normalized columns tensor with variance of output being square of std."""
    output = torch.randn(weights.size())
    output *= std / torch.sqrt(output.pow(2).sum(1).expand_as(output))
    return output


def weights_init(nn):
    """Initializes weights of the neural network nn based on optimal rates of learning."""
    class_name = nn.__class__.__name__
    if class_name.find('Conv') != -1:  # for the convolution layers
        weight_shape = list(nn.weight.data.size())
        fan_in = numpy.prod(weight_shape[1:4])
        fan_out = numpy.prod(weight_shape[2:4])*weight_shape[0]
        w_bound = numpy.sqrt(6. / fan_in+fan_out)
        nn.weight.data.uniform_(-w_bound, w_bound)
        nn.bias.data.fill_(0)
    elif class_name.find('Linear') != -1:  # for full connection layers
        weight_shape = list(nn.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = numpy.sqrt(6. / fan_in + fan_out)
        nn.weight.data.uniform_(-w_bound, w_bound)
        nn.bias.data.fill_(0)
