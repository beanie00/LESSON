import numpy as np
import utils
import time
import wandb


def set_log(
    self, num_frames, start_time, episode, return_per_frame_, test_return_per_frame_, current_option, option_termination_prob, option_termination_error, sigmoid_termonations
):
    return_per_episode = utils.synthesize(self.logs["rewards"])
    return_per_frame_.append(list(return_per_episode.values())[2])

    duration = int(time.time() - start_time)
    header = ["episode", "frames", "duration"]
    data = [episode, num_frames, duration]
    header += ["return_" + key for key in return_per_episode.keys()]
    data += return_per_episode.values()
    header += ["policy_loss"]
    data += [np.mean(self.logs["loss"])]
    header += ["Eval/test_return_sum"]
    data += [test_return_per_frame_[-1] if len(test_return_per_frame_) > 0 else 0]
    header += ["curr_option"]
    data += [current_option]
    if option_termination_prob and option_termination_error:
        header += ["Termination/option_termination_prob"]
        data += [option_termination_prob]
        header += ["Termination/option_termination_error"]
        data += [option_termination_error]
    if sigmoid_termonations and len(sigmoid_termonations) == 4:
        header += ["Termination/termination-random", "Termination/termination-z", "Termination/termination-rnd", "Termination/termination-e"]
        data += [sigmoid_termonations[0], sigmoid_termonations[1], sigmoid_termonations[2], sigmoid_termonations[3]]

    self.txt_logger.info(
        "E {} | F {:06} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} | pL {:.3f}".format(*data)
    )

    if self.log_wandb:
        for field, value in zip(header, data):
            wandb.log({field: value}, step=num_frames)
