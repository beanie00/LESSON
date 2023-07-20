import numpy as np
import collections

class EpisodeMemory:
    """Episode memory for recurrent agent"""

    def __init__(self, max_epi_num=100, max_epi_len=100):
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len

        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self, preprocess_obs, device):
        idx = np.random.randint(0, len(self.memory))
        sample = self.memory[idx].sample()

        steps = sample["step"]
        obs = preprocess_obs(sample["obs"], device=device)
        new_obs = preprocess_obs(
            sample["new_obs"], device=device
        )
        actions = sample["acts"]
        rewards = sample["rews"]
        rewards_i = sample["rews_i"]
        dones = sample["done"]
        options = sample["option"]

        collected_experience = {
            "steps": steps,
            "obs": obs,
            "new_obs": new_obs,
            "actions": actions,
            "rewards": rewards,
            "rewards_i": rewards_i,
            "dones": dones,
            "options": options
        }
        return collected_experience  # buffers, sequence_length

    def __len__(self):
        return len(self.memory)
