# _*_ coding: utf-8 _*_
import torch
import torch.nn as nn
import math
import numpy as np
from thop.vision.calc_func import (calculate_conv2d_flops, calculate_linear, calculate_adaptive_avg, calculate_norm)

__all__ = [
    "sub_filter_start_end",
    "get_same_padding",
    "weight_standardization",
    "min_divisible_value",
    "make_divisible",
    "build_activation",
    "copy_bn",
    "rm_bn_from_net",
    "adjust_bn_according_to_idx",
    "val2list",
    "multi_acc",
    "calc_learning_rate",
    "build_optimizer",
    "Merge",
    "list_sum",
    "list_mean",
    "list_stand_mean",
    "split",
    "list2str",
    "int2list",
    "get_macs"
]


def val2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def weight_standardization(weight, WS_EPS):
    if WS_EPS is not None:
        weight_mean = (
            weight.mean(dim=1, keepdim=True)
                .mean(dim=2, keepdim=True)
        )
        weight = weight - weight_mean
        std = (
                weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1)
                + WS_EPS
        )
        weight = weight / std.expand_as(weight)
    return weight


def adjust_bn_according_to_idx(bn, idx):
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    if type(bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
        bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)


def copy_bn(target_bn, src_bn):
    feature_dim = (
        target_bn.num_channels
        if isinstance(target_bn, nn.GroupNorm)
        else target_bn.num_features
    )

    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])
    if type(src_bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
        target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])


def min_divisible_value(n1, v1):
    """make sure v1 is divisible by n1, otherwise decrease v1"""
    if v1 >= n1:
        return n1
    while n1 % v1 != 0:
        v1 -= 1
    return v1


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def build_activation(act_func, inplace=True):
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_func == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func == "h_swish":
        return nn.Hardswish()
    elif act_func == "h_sigmoid":
        return nn.Hardsigmoid()
    elif act_func is None or act_func == "none":
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


def copy_bn(target_bn, src_bn):
    feature_dim = (
        target_bn.num_channels
        if isinstance(target_bn, nn.GroupNorm)
        else target_bn.num_features
    )

    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])
    if type(src_bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
        target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])


def rm_bn_from_net(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.forward = lambda x: x


def adjust_bn_according_to_idx(bn, idx):
    bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
    bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
    if type(bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
        bn.running_mean.data = torch.index_select(bn.running_mean.data, 0, idx)
        bn.running_var.data = torch.index_select(bn.running_var.data, 0, idx)


def calc_learning_rate(
        epoch, init_lr, n_epochs, batch=0, nBatch=None, lr_schedule_type="cosine"
):
    if lr_schedule_type == "cosine":
        t_total = n_epochs * nBatch
        t_cur = epoch * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type == "cosine_restart":
        lr_min = 0.0000000001
        t_cur = epoch * nBatch + batch
        t_0 = 6 * nBatch
        lr = lr_min + 0.5 * (init_lr - lr_min) * (1 + math.cos(math.pi * t_cur / t_0))

    elif lr_schedule_type is None:
        lr = init_lr

    elif lr_schedule_type == "None":
        lr = init_lr
    else:
        raise ValueError("do not support: %s" % lr_schedule_type)
    return lr


""" optimizer """


def build_optimizer(
        net_params, opt_type, opt_param, init_lr, weight_decay, no_decay_keys
):
    if no_decay_keys is not None:
        assert isinstance(net_params, list) and len(net_params) == 2
        net_params = [
            {"params": net_params[0], "weight_decay": weight_decay},
            {"params": net_params[1], "weight_decay": 0},
        ]
    else:
        net_params = [{"params": net_params, "weight_decay": weight_decay}]

    if opt_type == "sgd":
        opt_param = {} if opt_param is None else opt_param
        momentum, nesterov = opt_param.get("momentum", 0.9), opt_param.get(
            "nesterov", True
        )
        optimizer = torch.optim.SGD(
            net_params, init_lr, momentum=momentum, nesterov=nesterov
        )
    elif opt_type == "adam":
        optimizer = torch.optim.Adam(net_params, init_lr)
    else:
        raise NotImplementedError
    return optimizer


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def list_sum(x):
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x):
    return list_sum(x) / len(x)


def list_stand_mean(x):
    x = np.array(x)
    max_ = np.max(x)
    min_ = np.min(x)
    x = (x - min_) / (max_ - min_)
    return list_mean(x.to_list())


def multi_acc(acc, param, gamma=2):
    param = np.array(param)
    p = (param.max() - param.min()) * 100 * gamma
    acc = np.array(acc)
    print("P=%f" % p)
    return acc.mean() + p


def sort_settings(settings, params):
    indices = np.argsort(params)
    return indices


def split(category, search_keys, indexs, target):
    slot = dict()
    for k, index in zip(search_keys, indexs):
        slot[k] = category[target][index[0]:index[1]]
    return slot


def int2list(x):
    if isinstance(x, list):
        return x
    return [x]


def list2str(l):
    s = ''
    for _ in l:
        s += str(_) + '-'
    return s[:-1]


def get_macs(m):
    macs = 0
    if not hasattr(m, "meta_info"):
        return macs
    x = m.meta_info["x"]
    y = m.meta_info["y"]
    if isinstance(m, nn.Conv1d):
        macs += calculate_conv2d_flops(
            input_size=list(x),
            output_size=list(y),
            kernel_size=list(m.weight.shape),
            groups=m.groups,
            bias=m.bias
        )
    elif isinstance(m, nn.Linear):
        macs += calculate_linear(m.in_features, m.out_features)

    elif isinstance(m, nn.AdaptiveAvgPool1d):
        kernel = torch.div(
            torch.DoubleTensor([*(x[2:])]),
            torch.DoubleTensor([*(y[2:])])
        )
        total_add = torch.prod(kernel)
        num_elements = l_prod(y)
        macs += calculate_adaptive_avg(total_add, num_elements)

    elif isinstance(m, nn.BatchNorm1d):
        # bn is by default fused in inference
        flops = calculate_norm(l_prod(x))
        if m.affine:
            flops *= 2
        macs += flops
    else:
        raise NotImplementedError("Don't have the test program for", m.__repr__)
    return macs


def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res
