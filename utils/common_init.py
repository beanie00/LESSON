import torch
import wandb

import utils
from rl_algorithm.common.option_model import OptionQ
from rl_algorithm.dqn.config import device


def get_max_episode_length(env):
    if "MultiRoom-N2-S4" in env:
        return 40
    elif "MultiRoom-N4-S5" in env:
        return 80
    elif "PutNear-6x6" in env:
        return 30
    else:
        return 100

def init(self, env, preprocess_obs, args, train_interval=4):
    self.device = device
    self.preprocess_obs = preprocess_obs
    self.n_actions = env.action_space.n

    self.learn_step_counter = 1
    self.rnd_learn_step_counter = 1

    # train, test, logging parameter
    self.max_episode_length = get_max_episode_length(args.env)
    self.update_target_per_train = 100
    self.train_interval = train_interval
    self.rnd_train_interval = train_interval
    self.test_interval = 5000
    self.log_interval = 500
    self.algorithm = args.algorithm
    self.softmax_ww = args.softmax_ww

    # optimizer
    self.optimizer = torch.optim.RMSprop(self.policy_network.parameters(), args.lr)

    if self.log_wandb:
        wandb.config.update(
            {
                "env": args.env,
                "algorithm": args.algorithm,
                "max_episode_length": self.max_episode_length,
                "train_interval": self.train_interval,
                "rnd_train_interval": self.rnd_train_interval,
                "seed": args.seed,
                "softmax_ww": self.softmax_ww
            }
        )

def init_rnd(self, args):
    self.rnd_scale = args.rnd_scale
    self.rnd_optimizer = torch.optim.RMSprop(self.rnd_policy_network.parameters(), args.lr)
    if self.log_wandb:
        wandb.config.update(
        {
            "rnd_scale": self.rnd_scale,
        })

def init_optionQ(self, env, args, exploration_options):
    obs_space, _ = utils.get_obss_preprocessor(env.observation_space)
    self.option_policy_network = OptionQ(obs_space, len(exploration_options), False).to(device)
    self.option_target_network = OptionQ(obs_space, len(exploration_options), False).to(device)

    self.exploration_options = exploration_options
    self.exploration_idx = 0
    self.option_optimizer = torch.optim.RMSprop(self.option_policy_network.parameters(), args.lr)

def init_log(self, model_dir):
    self.txt_logger = utils.get_txt_logger(model_dir)

    self.logs = {}
