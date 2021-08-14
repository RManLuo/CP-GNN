#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/8/5 16:31
# @Author  : Raymound luo
# @Mail    : luolinhao1998@gmail.com
# @File    : earlystop.py
# @Software: PyCharm
# @Describe:
import datetime
import numpy as np
import torch
import os
from utils.helper import save_config


class EarlyStopping(object):
    def __init__(self, checkpoint_path, config=None, patience=10):
        dt = datetime.datetime.now()
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        self.filepath = os.path.join(checkpoint_path, 'early_stop_{}_{:02d}-{:02d}-{:02d}'.format(
            dt.date(), dt.hour, dt.minute, dt.second))
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
        self.filename = os.path.join(self.filepath, "model.pth")
        if config:
            save_config(config, self.filepath)
        self.patience = patience
        self.counter = 0

        self.best_loss = None
        self.early_stop = False

    def step(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss >= self.best_loss):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss < self.best_loss):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
        return model
