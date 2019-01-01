"""
File for training the A3C agent implemented for the Breakout game.
Coded by: Sultan Sidhu
January 1, 2019
"""

import torch
import torch.nn.functional as func
from torch.autograd import Variable
from envs import create_atari_env
from model import ActorCritic


def ensure_shared_gradients(model, shared_model):
    """A method to ensure that the gradients are shared."""
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param is not None:
            return
        shared_param.grad = param.grad


def train(rank, params, shared_model, optimizer):
    """Trains the A3C model."""
    torch.manual_seed(params.seed + rank)  # shifts seed with the rank to desynchronize training agents.
    env = create_atari_env(params.env_name)
    env.seed(params.seed + rank)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    state = env.reset()  # gives a numpy array of dimensions 1*42*42, with 42*42 being the image size
    state = torch.from_numpy(state)
    done = True
