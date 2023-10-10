# _*_ coding: utf-8 _*_
import json
import numpy as np
import copy

KERNEL_IDX_TABLE = {3: 0, 5: 1, 7: 2}
EXPAND_RATIO_IDX_TABLE = {3: 0, 4: 1, 6: 2}
IDX_KERNEL_TABLE = {0: 3, 1: 5, 2: 7}
IDX_EXPAND_RATIO_TABLE = {0: 3, 1: 4, 2: 6}
x = 0
BIAS = [x]
for i in range(1, 10):
    x += 3 ** i
    BIAS.append(x)


def decode(config, deleted_keys=[]):
    category = {}

    def read(cfg):
        for k, v in cfg.items():
            if isinstance(v, dict):
                read(v)
                continue
            if k in deleted_keys:
                continue
            if k in category.keys():
                category[k].append(v)
            else:
                category[k] = [v]

    read(config)
    return category


def encode(category, config):

    def read(cfg):
        for k, v in cfg.items():
            if isinstance(v, dict):
                cfg[k] = read(v)
                continue
            if k not in category.keys():
                continue
            cfg[k] = category.get(k).pop(0)
        return cfg

    config = read(config)
    return config


def T2D(x, key="kernel"):
    # x=[3,3,3,3,3]
    if key == "kernel":
        table = KERNEL_IDX_TABLE
    else:
        table = EXPAND_RATIO_IDX_TABLE

    L = len(x)
    bias = BIAS[L-1]

    base = 0
    x = x[::-1]
    for i in range(L):
        base += table[x[i]] * (3 ** i)
    return bias + base


def D2T(x, key="kernel", L=4):
    if key == "kernel":
        table = IDX_KERNEL_TABLE
    else:
        table = IDX_EXPAND_RATIO_TABLE
    for i in range(len(BIAS)):
        if BIAS[i] > x:
            bias = BIAS[i-1]
            break
    x = x - bias
    result = []
    while True:
        result.append(table[x % 3])
        x = x // 3
        if x == 0:
            while len(result) < L:
                result.append(table[x % 3])
            break
    return result[::-1]

