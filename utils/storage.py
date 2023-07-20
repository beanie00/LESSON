import os
import logging
import sys

import utils


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

def get_storage_dir():
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    return "storage"

def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger()

def get_model_dir(model_name):
    return os.path.join(get_storage_dir(), model_name)