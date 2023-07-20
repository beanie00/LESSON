import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rl_algorithm.drqn.config import discount, device, embedding_size, rnn_hidden_dim
from rl_algorithm.common.base_model import BaseModel


class DRQN(BaseModel):
    def __init__(self, obs_space, action_space, device, args):
        super().__init__(obs_space)
        self.args = args
        self.device = device
        self.Linear1 = nn.Linear(embedding_size, rnn_hidden_dim)
        self.lstm = nn.LSTM(rnn_hidden_dim, rnn_hidden_dim, batch_first=True)
        self.Linear2 = nn.Linear(rnn_hidden_dim, action_space.n)

    def init_hidden_state(self, training=None):
        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            hidden_state = torch.zeros(
                [1, rnn_hidden_dim]
            ), torch.zeros([1, rnn_hidden_dim])
            return hidden_state
        else:
            return torch.zeros([1, rnn_hidden_dim]), torch.zeros([1, rnn_hidden_dim])

    def forward(self, obs, hidden_state):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = x.reshape(x.shape[0], -1)
        x = self.image_conv(x)
        x = F.relu(self.Linear1(x))
        x, new_hidden_state = self.lstm(
            x, (hidden_state[0].to(self.device), hidden_state[1].to(self.device))
        )
        x = self.Linear2(x)
        return x, new_hidden_state

    @classmethod
    def train_model(
        cls,
        online_net,
        target_net,
        optimizer,
        collected_experience,
        is_rnd=False,
    ):
        obs = collected_experience["obs"]
        new_obs = collected_experience["new_obs"]
        actions = collected_experience["actions"]
        rewards = collected_experience["rewards"]
        rewards_i = collected_experience["rewards_i"]
        dones = collected_experience["dones"]
        indices = np.arange(len(collected_experience["actions"]))

        if is_rnd:
            rewards = rewards_i

        hidden_state_target = target_net.init_hidden_state(training=True)

        q_target, _ = target_net(new_obs, hidden_state_target)
        max_actions = q_target.max(dim=1)[0]

        Q_target = torch.tensor(rewards, device=device) + discount * max_actions * torch.tensor(dones, device=device)

        hidden_state = online_net.init_hidden_state(training=True)
        q_policy, _ = online_net(obs, hidden_state)

        Q_policy = q_policy[indices, actions]

        # compute loss
        loss = nn.functional.smooth_l1_loss(Q_policy, Q_target)

        # Update Network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss
