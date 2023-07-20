import torch.nn as nn
from rl_algorithm.dqn.config import device

class BaseModel(nn.Module):
    def __init__(self, obs_space):
        super().__init__()

        n = obs_space["image"][0]
        m = obs_space["image"][1]
        z = obs_space["image"][2]
        shape = n*m*z
        self.image_conv = nn.Sequential(
            nn.Linear(shape, shape//2),
            nn.ReLU(),
            nn.Linear(shape//2, shape//4),
            nn.ReLU(),
            nn.Linear(shape//4, 64),
            nn.ReLU()
        ).to(device)