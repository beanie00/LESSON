import random
import numpy
import torch
import collections


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["sum"] = numpy.sum(array)
    return d


class DictList(dict):
    """
    A dictionnary of lists of same size
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value
