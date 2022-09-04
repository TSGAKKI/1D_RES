from contextlib import contextmanager
from sklearn.metrics import precision_recall_curve, accuracy_score, roc_auc_score
from sklearn.metrics import f1_score, recall_score, precision_score
from collections import OrderedDict, defaultdict
from itertools import repeat
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from scipy.sparse import linalg
import sklearn
import matplotlib.cm as cm
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import math
import tqdm
import shutil
import queue
import random
import time
import json
import torch
import h5py
import logging
import numpy as np
import os
import sys
import pickle
import scipy.sparse as sp


def seed_torch(seed=123):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_save_dir(base_dir, training, id_max=500):
    """
    Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).
    Args:
        base_dir (str):     Base directory in which to make save directories
        training (bool):    Save dir. is for training (determines subdirectory).
        id_max (int):       Maximum ID number before raising an exception.
    Returns:
        save_dir (str):     Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, '{}-{:02d}'.format(subdir, uid))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger

def count_parameters(model):
    """
    Counter total number of parameters, for Pytorch
    """
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_param = 0
   
    for param_tensor in model.state_dict():
        # print(param_tensor, '\t', net.state_dict()[param_tensor].size(), flush=True)
        total_param += np.prod(model.state_dict()[param_tensor].size())
    return total_param

def load_model_checkpoint(checkpoint_file, model, optimizer=None):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        return model, optimizer

    return model

class CheckpointSaver:
    """Class to save and load model checkpoints.
    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.
    Args:
        save_dir (str): Directory to save checkpoints.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """

    def __init__(self, save_dir, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        
    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.
        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        if self.best_val <= metric_val:
            return False
        else:
            return True

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, epoch, model, optimizer, metric_val):
        """Save model parameters to disk.
        Args:
            epoch (int): Current epoch.
            model (torch.nn.DataParallel): Model to save.
            optimizer: optimizer
            metric_val (float): Determines whether checkpoint is best so far.
        """
        ckpt_dict = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }

        checkpoint_path = os.path.join(self.save_dir, 'last.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)

        best_path = ''
        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            # self._print('New best checkpoint at epoch {}...'.format(epoch))

class DominGenerator():
    '''
    Domin Generator
    '''
    k = -1       # the fold number
    l_list = []  # length of each domin
    d_list = []  # d list with length=k

    # Initializate
    def __init__(self, len_list):
        self.l_list = len_list
        self.k = len(len_list)

    # Get i-th fold
    def getFold(self, i):
        isFirst = True
        isFirstVal = True
        j = 0   #1~9
        ii = 0  #1~10
        for l in self.l_list:
            if ii != i:
                a = np.zeros((l, 9), dtype=int)
                a[:, j] = 1
                if isFirst:
                    train_domin = a
                    isFirst = False
                else:
                    train_domin = np.concatenate((train_domin, a))
                j += 1
            else:
                if isFirstVal:
                    val_domin = np.zeros((l, 9), dtype=int)
                    isFirstVal = False
                else:
                    a = np.zeros((l, 9), dtype=int)
                    val_domin = np.concatenate((val_domin, a))
            ii += 1
        return train_domin, val_domin