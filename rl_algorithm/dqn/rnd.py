import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from rl_algorithm.dqn.config import embedding_size
from rl_algorithm.common.base_model import BaseModel

class RunningStats:
    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())


class NN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_hid):
        super(NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hid = n_hid

        self.fc1 = torch.nn.Linear(in_dim, n_hid, "linear")
        self.fc2 = torch.nn.Linear(n_hid, out_dim, "linear")
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        return y


class RND(BaseModel):
    def __init__(self, obs_space, out_dim, n_hid, device):
        super().__init__(obs_space)
        self.target = NN(embedding_size, out_dim, n_hid).to(device)
        self.model = NN(embedding_size, out_dim, n_hid).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.reward_i_rms = RunningStats()
        self.device = device

    def get_reward(self, obs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = x.reshape(x.shape[0], -1)
        x = self.image_conv(x)
        y_true = self.target(x)
        y_pred = self.model(x)

        reward = torch.pow(y_pred - y_true.detach(), 2).mean()

        self.reward_i_rms.push(reward.detach().item())
        reward = (reward - self.reward_i_rms.mean()) / (self.reward_i_rms.standard_deviation() + 1e-10)

        return reward

    def update(self, Ri):
        loss = Ri.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
