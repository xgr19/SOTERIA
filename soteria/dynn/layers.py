# _*_ coding: utf-8 _*_
import torch.nn as nn
import soteria.dynn as dynn
from soteria.utils import get_macs

__all__ = ["Dynamic_Layer",
           "Sequential",
           "ModuleList"]


class Dynamic_Layer(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    @property
    def add_config(self):
        return {}, False

    @property
    def config(self):
        cfg, cover = self.add_config
        if cover:
            return cfg
        for name, module in self.named_children():
            if isinstance(module, dynn.Dynamic_Module) or \
                    isinstance(module, Dynamic_Layer):
                if isinstance(module, dynn.BatchNorm1d):
                    continue
                cfg[name] = module.config
        return cfg

    def set_active(self, config):
        for k, v in config.items():
            if isinstance(v, dict):
                self.__getattr__(k).set_active(v)
            else:
                self.__setattr__(k, v)

    def get_active_module(self):
        for name, child in self.named_children():
            if isinstance(child, dynn.Dynamic_Module):
                active_module = self.__getattr__(name).get_active_module()
                self.__setattr__(name, active_module)
            elif isinstance(child, Dynamic_Layer):
                child.get_active_module()

    def get_profile(self):
        macs = 0
        for name, child in self.named_children():
            if isinstance(child, dynn.Dynamic_Layer):
                macs += child.get_profile()
            else:
                macs += get_macs(child)
        return macs

    def get_parameters(self, keys=None, mode="include"):
        if keys is None:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    yield param
        elif mode == "include":
            for name, param in self.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag and param.requires_grad:
                    yield param
        elif mode == "exclude":
            for name, param in self.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag and param.requires_grad:
                    yield param
        else:
            raise ValueError("do not support: %s" % mode)

    def weight_parameters(self):
        return self.get_parameters()


class Sequential(nn.Sequential, Dynamic_Layer):
    def __init__(self, args):
        super(Sequential, self).__init__(args)


class ModuleList(nn.ModuleList, Dynamic_Layer):
    def __init__(self, modules):
        super(ModuleList, self).__init__(modules)

