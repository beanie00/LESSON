import numpy as np
from typing import Dict

class EpisodeBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.step = []
        self.obs = []
        self.action = []
        self.reward = []
        self.new_obs = []
        self.done = []
        self.reward_i = []
        self.option = []

    def put(self, transition):
        self.step.append(transition["step"])
        self.obs.append(transition["obs"])
        self.action.append(transition["action"])
        self.reward.append(transition["reward"])
        self.reward_i.append(transition["reward_i"])
        self.new_obs.append(transition["new_obs"])
        self.done.append(transition["done"])
        self.option.append(transition["option"])

    def sample(
        self
    ) -> Dict[str, np.ndarray]:
        step = np.array(self.step)
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        reward_i = np.array(self.reward_i)
        new_obs = np.array(self.new_obs)
        done = np.array(self.done)
        option = np.array(self.option)

        return dict(
            step=step,
            obs=obs,
            acts=action,
            rews=reward,
            rews_i=reward_i,
            new_obs=new_obs,
            done=done,
            option=option
        )

    def __len__(self) -> int:
        return len(self.obs)
