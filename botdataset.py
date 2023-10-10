# !/usr/bin/env python
# -*-coding:utf-8 -*-
from torch.utils.data import Dataset
import pickle


class BotDataset(Dataset):
    def __init__(self, path, key="train"):
        super(BotDataset, self).__init__()
        with open(path, "rb") as f:
            self.X, self.y  = pickle.load(f)[key]
    def __len__(self):
        return len(self.y)
    def __getitem__(self, item):
        x = self.X[item] # a vector, the input features of a sample X.
        y = self.y[item]
        return x.astype("float32"), y
