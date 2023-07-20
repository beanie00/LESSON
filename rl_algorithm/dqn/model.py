import torch
import torch.nn as nn
import numpy as np

from rl_algorithm.dqn.config import embedding_size, word_embedding_size, text_embedding_size, discount, batch_size, device
from rl_algorithm.common.base_model import BaseModel

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1,
                                                                 keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class DQN(BaseModel):
    def __init__(self, obs_space, action_space, is_init, is_use_mission):
        super().__init__(obs_space)
        total_embedding_size = embedding_size
        self.is_use_mission = is_use_mission

        if is_use_mission:
            self.word_embedding = nn.Embedding(obs_space["text"], word_embedding_size)
            self.text_rnn = nn.GRU(word_embedding_size, text_embedding_size, batch_first=True)
            total_embedding_size += text_embedding_size

        self.actor = nn.Sequential(
            nn.Linear(total_embedding_size, embedding_size),
            nn.Tanh(),
            nn.Linear(embedding_size, action_space.n)
        )
        if is_init:
            self.apply(init_params)

    def forward(self, obs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = x.reshape(x.shape[0], -1)
        x = self.image_conv(x)

        if self.is_use_mission:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((x, embed_text), dim=1)
        else:
            embedding = x

        x = self.actor(embedding)

        return x

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

    @classmethod
    def train_model(
        cls, online_net, target_net, optimizer, collected_experience, is_rnd
    ):
        obs = collected_experience["obs"]
        new_obs = collected_experience["new_obs"]
        actions = collected_experience["actions"]
        rewards = collected_experience["rewards"]
        rewards_i = collected_experience["rewards_i"]
        dones = collected_experience["dones"]
        indices = np.arange(batch_size)

        if is_rnd:
            rewards = torch.tensor(rewards_i, device=device) * 10
        else:
            rewards = torch.tensor(rewards, device=device)

        Q_policy = online_net(obs)[indices, actions]
        max_actions = target_net(new_obs).max(dim=1)[0]

        Q_target = rewards + discount * max_actions * torch.tensor(dones, device=device)

        loss = nn.functional.smooth_l1_loss(Q_policy, Q_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss