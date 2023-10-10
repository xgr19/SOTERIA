# _*_ coding: utf-8 _*_
import pickle
import os
from .base_manager import Manager
from .parser import decode, encode
from soteria.utils import count_parameters, count_net_flops, measure_net_latency
import torch
import random
import numpy as np
import copy
torch.manual_seed(2022)
torch.cuda.manual_seed_all(2022)


class RdManager(Manager):
    def __init__(self, net, train_dataset, test_dataset, cfg):
        super(RdManager, self).__init__(net, train_dataset, test_dataset, cfg)
        if cfg.get("re_organize_middle_weights", False):
            self.net.re_organize_middle_weights()
            self.write_log("re_organize_middle_weights")
        # ----------Init----------#
        self.num_sample_train_per_epoch = cfg.num_sample_train_per_epoch
        self.num_sample_val_per_epoch = cfg.num_sample_val_per_epoch

        self.object_type = cfg.object_type
        self.input_shape = cfg.input_shape

        self.deleted_keys = self.cfg.deleted_keys
        max_net_setting = self.net.config
        self.max_net_setting = max_net_setting
        self.decode = lambda config: decode(config, self.deleted_keys)
        self.encode = lambda category: encode(category, copy.deepcopy(max_net_setting))
        self.category = self.decode(self.max_net_setting) # 提前写好模板
        self.val_record = np.zeros((self.num_sample_val_per_epoch, 2))
        self.train_record = np.zeros(self.num_sample_train_per_epoch)

    def random_sample(self, search_space=None):
        if search_space is None:
            search_space = self.cfg.search_space
        category = copy.copy(self.category)
        for k, v in search_space.items():
            category[k] = [random.choice(v) for _ in range(len(category[k]))]
        return self.encode(category)

    def profile(self, static_net):
        if self.object_type == "flops":
            obj = count_net_flops(static_net, [1] + list(self.input_shape)) / 1e6
        elif self.object_type == "latency":
            obj, _  = measure_net_latency(static_net, "cpu", fast=False, input_shape=self.input_shape)
        else:
            obj = count_parameters(static_net) / 1e6
        return obj

    def get_settings(self, epoch = 0):
        self.val_record = np.zeros((self.num_sample_val_per_epoch, 2))
        self.train_record = np.zeros(self.num_sample_train_per_epoch)
        train_settings = [self.random_sample() for _ in range(self.num_sample_train_per_epoch)]
        val_settings = train_settings[:self.num_sample_val_per_epoch]
        return train_settings, val_settings

    def callback_train(self, Acc, settings, epoch=0):
        num = 0
        for acc in Acc:
            self.train_record[num] = 100 - acc
            num += 1
            if num == self.num_sample_train_per_epoch:
                break
        path = self.record_path + "/%d-train.pkl" % epoch
        with open(path, 'wb') as f:
            pickle.dump((self.train_record, settings), f)
        self.write_log(path)

    def callback_val(self, Acc, Obj, epoch=0):
        num = 0
        for acc, obj in zip(Acc, Obj):
            self.val_record[num, 0] = 100 - acc
            self.val_record[num, 1] = obj
            num += 1
        path = self.record_path + "/%d-val.pkl" % epoch
        with open(path, 'wb') as f:
            pickle.dump(self.val_record, f)
        self.write_log(path)

    @property
    def record_path(self):
        if self.__dict__.get("_record_path", None) is None:
            logs_path = os.path.join(self.path, "record")
            os.makedirs(logs_path, exist_ok=True)
            self.__dict__["_record_path"] = logs_path
        return self.__dict__["_record_path"]









