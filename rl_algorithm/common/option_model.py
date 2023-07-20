import torch
import torch.nn as nn
import numpy as np

from rl_algorithm.dqn.config import embedding_size, discount, device
from rl_algorithm.common.base_model import BaseModel

def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

def softmax(a, ww):
    c = max(np.max(a), 1)
    exp_a = np.exp(np.array(a)/c * ww)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    y = np.clip(y, 0.05, 0.95)
    return y


class OptionQ(BaseModel):
    def __init__(self, obs_space, num_options, is_init):
        super().__init__(obs_space)

        self.actor = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.Tanh(),
            nn.Linear(embedding_size, num_options)
        )
        self.terminations = nn.Linear(embedding_size, num_options)
        if is_init:
            self.apply(init_params)

    def preprocess_obs(self, obs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = x.reshape(x.shape[0], -1)
        return self.image_conv(x)

    def forward(self, obs):
        processed_obs = self.preprocess_obs(obs)
        x = self.actor(processed_obs)

        return x

    def get_td_error(online_net, target_net, collected_experience, algorithm):
        obs = collected_experience["obs"]
        new_obs = collected_experience["new_obs"]
        actions = collected_experience["actions"]
        rewards = collected_experience["rewards"]
        dones = collected_experience["dones"]
        indices = np.arange(len(collected_experience["actions"]))

        if algorithm == 'dqn':
            q_target= target_net(new_obs)
            q_policy = online_net(obs)
        if algorithm == 'drqn':
            hidden_state_target = target_net.init_hidden_state(training=True)
            q_target, _ = target_net(new_obs, hidden_state_target)
            hidden_state = online_net.init_hidden_state(training=True)
            q_policy, _ = online_net(obs, hidden_state)

        max_actions = q_target.max(dim=1)[0]
        Q_target = torch.tensor(rewards, device=device) + discount * max_actions * torch.tensor(dones, device=device)

        Q_policy = q_policy[indices, actions]

        return abs(Q_target - Q_policy)

    @classmethod
    def get_option_td_error(cls, self, collected_experience):
        obs = collected_experience["obs"]
        new_obs = collected_experience["new_obs"]
        options = collected_experience["options"]
        done_masks = collected_experience["dones"]
        rewards = np.array(list(map(float, collected_experience["rewards"])))
        rewards_i = np.array(list(map(float, collected_experience["rewards_i"])))
        indices = np.arange(len(obs))

        rewards_i = rewards_i * self.rnd_scale
        rewards = np.add(rewards, rewards_i)

        next_Q_target = self.option_target_network(new_obs)
        next_termination_probs = self.option_policy_network.get_terminations(new_obs).detach()
        next_options_term_prob = torch.clip(next_termination_probs[indices, options], 0.1, 0.9)

        torch_rewards = torch.tensor(np.array(rewards), device=device)

        gt = torch_rewards + torch.tensor(np.array(done_masks), device=device) * discount * \
        ((1 - next_options_term_prob) * next_Q_target[indices, options] + next_options_term_prob  * next_Q_target.max(dim=-1)[0])

        Q_policy = self.option_policy_network(obs)

        # compute loss
        td_err = (Q_policy[indices, options] - gt.detach()).pow(2).mul(0.5).mean()
        return td_err

    @classmethod
    def get_termination_loss_batch(cls, self, collected_experience):
        obs = collected_experience["obs"]
        options = collected_experience["options"]
        done_masks = collected_experience["dones"]
        done_masks = torch.tensor(done_masks, device=device)
        indices = np.arange(len(obs))

        option_term_prob = self.option_policy_network.get_terminations(obs)[indices, options]
        Q_target = self.option_target_network(obs)
        error_before_reg = Q_target[indices, options].detach() - Q_target.max(dim=1)[0].detach()
        reg = - 1 * error_before_reg.mean()
        termination_error = Q_target[indices, options].detach() - Q_target.max(dim=1)[0].detach() + reg
        termination_loss = option_term_prob * termination_error * done_masks
        return termination_loss.mean(), termination_error.mean()

    def predict_option_termination(self, obs, current_option):
        processed_obs = self.preprocess_obs(obs)
        termination = self.terminations(processed_obs)[:, current_option].sigmoid()
        sigmoid_termonations = [self.terminations(processed_obs)[:, o].sigmoid().item() for o in range(4)]

        termination = torch.clip(termination, 0.1, 0.9)
        sigmoid_termonations = np.clip([self.terminations(processed_obs)[:, o].sigmoid().item() for o in range(4)], 0.1, 0.9).tolist()

        return termination, sigmoid_termonations

    def get_terminations(self, obs):
        processed_obs = self.preprocess_obs(obs)
        return self.terminations(processed_obs).sigmoid()

    def choice(self, options,probs):
        x = np.random.rand()
        cum = 0
        for i,p in enumerate(probs):
            cum += p
            if x < cum:
                break
        return options[i]

    def select_option(self, obs, exploration_options, ww):
        processed_obs = self.preprocess_obs(obs)
        Q = self.actor(processed_obs)

        exploration_ratio = Q.tolist()[0]
        exploration_ratio = softmax(exploration_ratio, ww)
        # next_option = np.random.choice(
            # len(exploration_options), 1, p=exploration_ratio
        # )[0]
        next_option = self.choice([0, 1, 2, 3], exploration_ratio)

        return next_option