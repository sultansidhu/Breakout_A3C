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


def weights_init(n):
    """Initializes weights of the neural network nn based on optimal rates of learning."""
    class_name = n.__class__.__name__
    if class_name.find('Conv') != -1:  # for the convolution layers
        weight_shape = list(n.weight.data.size())
        fan_in = numpy.prod(weight_shape[1:4])
        fan_out = numpy.prod(weight_shape[2:4])*weight_shape[0]
        w_bound = numpy.sqrt(6. / fan_in+fan_out)
        n.weight.data.uniform_(-w_bound, w_bound)
        n.bias.data.fill_(0)
    elif class_name.find('Linear') != -1:  # for full connection layers
        weight_shape = list(n.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = numpy.sqrt(6. / fan_in + fan_out)
        n.weight.data.uniform_(-w_bound, w_bound)
        n.bias.data.fill_(0)

# Making the brain of the A3C model


class ActorCritic(torch.nn.Module):
    """ Class managing the actor-critic dynamic of the A3C agent. Initializes actor-critic and the LSTM cell. """

    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32*3*3, 256)  # enables the neural network to understand temporal relationships
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1)  # the V(s) for the state
        self.actor_linear = nn.Linear(256, num_outputs)   # the Q(s, a) for the state
        # initializing weights
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1)
        self.critic_linear.bias.data.fill_(0)
        # small std. deviation for actor and large for critic as defined above make a
        # good balance of exploration vs exploitation
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, inputs):
        """The forward propagation through the A3C neural network."""
        inputs, (hx, cx) = inputs
        x = func.elu(self.conv1(inputs))
        x = func.elu(self.conv2(x))
        x = func.elu(self.conv3(x))
        x = func.elu(self.conv4(x))
        x = x.view(-1, 32*3*3)
        (hx, cx) = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
