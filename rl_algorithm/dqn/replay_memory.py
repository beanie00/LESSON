import random
from rl_algorithm.dqn.config import device

class ReplayMemory(object):
    """
    Memory buffer for Experience Replay
    """

    def __init__(self, capacity, preprocess_obs):
        """
        Initialize a buffer containing max_size experiences
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.preprocess_obs = preprocess_obs

    def add(self, experience):
        """
        Add an experience to the buffer
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer
        """
        batch = random.sample(self.memory, batch_size)
        obs = self.preprocess_obs([exp["obs"] for exp in batch], device=device)
        new_obs = self.preprocess_obs(
            [exp["new_obs"] for exp in batch], device=device
        )
        actions = [exp["action"] for exp in batch]
        rewards = [exp["reward"] for exp in batch]
        rewards_i = [exp["reward_i"] for exp in batch]
        dones = [exp["done"] for exp in batch]
        options = [exp["option"] for exp in batch]

        collected_experience = {
            "obs": obs,
            "new_obs": new_obs,
            "actions": actions,
            "rewards": rewards,
            "rewards_i": rewards_i,
            "dones": dones,
            "options": options
        }
        return collected_experience
    def __len__(self):
        return len(self.memory)
