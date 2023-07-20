import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from rl_algorithm.drqn.config import embedding_size, rnn_hidden_dim, rnd_out_dim, device
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

class NN(BaseModel):
    def __init__(self, obs_space, args):
        super().__init__(obs_space)
        self.out_dim = rnd_out_dim

        self.Linear1 = nn.Linear(embedding_size, rnn_hidden_dim)
        self.lstm = nn.LSTM(rnn_hidden_dim, rnn_hidden_dim, batch_first=True)
        self.Linear2 = nn.Linear(rnn_hidden_dim, rnd_out_dim)

        self.args = args

    def init_hidden_state(self, training=None):
        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            hidden_state = torch.zeros(
                [1, rnn_hidden_dim]
            ), torch.zeros([1, rnn_hidden_dim])
            return hidden_state
        else:
            return torch.zeros([1, rnn_hidden_dim]), torch.zeros([1, rnn_hidden_dim])

    def forward(self, obs, hidden):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = x.reshape(x.shape[0], -1)
        x = self.image_conv(x)
        x = F.relu(self.Linear1(x))
        x, new_hidden = self.lstm(x, (hidden[0].to(device), hidden[1].to(device)))
        x = self.Linear2(x)
        return x, new_hidden


class RNNRND:
    def __init__(self, obs_space, args, device):
        self.device = device

        self.target = NN(obs_space, args).to(device)
        self.model = NN(obs_space, args).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        self.reward_i_rms = RunningStats()
        self.device = device

        self.hidden_state_target = self.target.init_hidden_state(training=False)
        self.hidden_state_model = self.model.init_hidden_state(training=False)

    def get_reward(self, x, training):
        if not training:
            hidden_state_target = self.hidden_state_target
            hidden_state_model = self.hidden_state_model
        else:
            hidden_state_target = self.target.init_hidden_state(training=True)
            hidden_state_model = self.model.init_hidden_state(training=True)

        y_true, _ = self.target(x, hidden_state_target)
        y_pred, new_hidden_state_model = self.model(x, hidden_state_model)


        self.hidden_state_model = new_hidden_state_model

        reward = torch.pow(y_pred - y_true.detach(), 2).mean()

        self.reward_i_rms.push(reward.detach().item())
        reward = (reward - self.reward_i_rms.mean()) / (self.reward_i_rms.standard_deviation() + 1e-10)

        return reward

    def update(self, Ri):
        loss = Ri.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
