import numpy as np
import logging
from datetime import datetime
import sys
import os
import shutil
import torch

def compute_weighted_f1(metric, target_names=None):
    # metric: sklearn.metrics.classification_report
    assert set(target_names).issubset(set(metric.keys()))

    supports = np.array([metric[name]["support"] for name in target_names])
    weights = supports / np.sum(supports)

    weighted_f1 = np.dot(weights, np.array([metric[name]["f1-score"] for name in target_names]))

    return weighted_f1

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
def prepare_logger(filename=None, name=None):
    name = __name__ if (name is None) or (not isinstance(name, str)) else name

    logger = logging.getLogger(name)
    if filename is None: filename = get_current_time()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s\n%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(filename)],
    )
    logger.setLevel(logging.INFO)

    return logger


def remove_directories(path_list):
    # path_list: a list of paths in the current directory
    for path in path_list:
        if os.path.exists(path): shutil.rmtree(path)





