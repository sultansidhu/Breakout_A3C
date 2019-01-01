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

    episode_length = 0
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        values = []
        log_probs = []
        rewards = []
        entropies = []
        for steps in range(params.num_steps):
            value, action_values, (hx, cx) = model((Variable(state.unsqueeze(0)), (hx, cx)))
            prob = func.softmax(action_values)
            log_prob = func.log_softmax(action_values)
            # the softmax and log softmax are applied to the obtained q values
            entropy = -(log_prob*prob).sum(1)
            entropies.append(entropy)
            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))
            values.append(value)
            log_probs.append(log_prob)
            state, reward, done, _ = env.step(action.numpy())
            done = (done or episode_length >= params.max_episode_length)
            reward = max(min(reward, 1), -1)
            if done:
                episode_length = 0
                state = env.reset()
            state = torch.from_numpy(state)
            rewards.append(reward)
            if done:
                break
        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hx, cx)))
            R = value.data
        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)  # generalized advantage estimation
        for i in reversed(range(len(rewards))):
            R = params.gamma*R+rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5*advantage.pow(2)
            TD = rewards[i] + params.gamma * values[i+1].data - values[i].data
            gae = gae*params.gamma*params.tau + TD
            policy_loss = policy_loss - log_probs[i]*Variable(gae) - 0.01*entropies[i]
        optimizer.zero_grad()
        (policy_loss + 0.5*value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_gradients(model, shared_model)
        optimizer.step()
