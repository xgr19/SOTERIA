# _*_ coding: utf-8 _*_
import torch
import copy
import time
import pandas as pd
import torch.nn as nn
import thop.profile as thopprofile
from soteria.utils import rm_bn_from_net
from mmt.parser import predict_latency

__all__ = ["profile",
           "get_net_device",
           "count_parameters",
           "count_net_flops",
           "measure_net_latency",
           "get_net_info",
           "measure_model"
           ]


def measure_model(net, input_shape, key="param", use_emb=False, task="unsw"):
    if key == "param":
        return count_parameters(net) / 1e+6
    elif key == "flops":
        return count_net_flops(net, data_shape=input_shape, use_emb=use_emb) / 1e+6
    elif key == "macs":
        return net.get_profile().numpy() / 1e+6
    return predict_latency(net, key, input_shape=[1] + input_shape)


def get_net_device(net):
    return net.parameters().__next__().device


def count_conv2d(m, _, y):
    cin = m.in_channels

    kernel_ops = m.weight.size()[2] * m.weight.size()[3]
    ops_per_element = kernel_ops
    output_elements = y.nelement()

    # cout x oW x oH
    total_ops = cin * output_elements * ops_per_element // m.groups
    m.total_ops = torch.zeros(1).fill_(total_ops)


def count_conv1d(m, _, y):
    cin = m.in_channels

    kernel_ops = m.weight.size()[2]
    ops_per_element = kernel_ops
    output_elements = y.nelement()
    # cout x oW x oH
    total_ops = cin * output_elements * ops_per_element // m.groups
    m.total_ops = torch.zeros(1).fill_(total_ops)


def count_linear(m, _, __):
    total_ops = m.in_features * m.out_features
    m.total_ops = torch.zeros(1).fill_(total_ops)


register_hooks = {
    nn.Conv1d: count_conv1d,
    nn.Conv2d: count_conv2d,
    ######################################
    nn.Linear: count_linear,
    ######################################
    nn.Dropout: None,
    nn.Dropout2d: None,
    nn.Dropout3d: None,
    nn.BatchNorm2d: None,
    nn.BatchNorm1d: None
}


def profile(model, input_size, custom_ops=None, use_emb=False):
    handler_collection = []
    custom_ops = {} if custom_ops is None else custom_ops

    def add_hooks(m_):
        if len(list(m_.children())) > 0:  # 不是叶子节点
            return

        m_.register_buffer("total_ops", torch.zeros(1))
        m_.register_buffer("total_params", torch.zeros(1))

        for p in m_.parameters():
            m_.total_params += torch.zeros(1).fill_(p.numel())

        m_type = type(m_)
        fn = None

        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]

        if fn is not None:
            _handler = m_.register_forward_hook(fn)
            handler_collection.append(_handler)

    original_device = get_net_device(model)
    # training = model.training
    model.to(original_device)
    model.eval()
    model.apply(add_hooks)
    if use_emb:
        x = torch.zeros(input_size).long().to(original_device)
    else:
        x = torch.zeros(input_size).to(original_device)

    with torch.no_grad():
        model(x)
    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue

        total_ops += m.total_ops
        total_params += m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()
    for handler in handler_collection:
        handler.remove()

    return total_ops, total_params


""" Network profiling """


def get_macs(net, data_shape=[3, 224, 224], use_emb=False):
    device = get_net_device(net)
    net.to(device)
    if use_emb:
        input = torch.ones([1] + data_shape).long().to(device)
    else:
        input = torch.ones([1] + data_shape).to(device)

    macs, _ = thopprofile(net, inputs=(input,))
    return macs


def get_net_device(net):
    return net.parameters().__next__().device


def count_parameters(net):
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_params


def count_net_flops(net, data_shape=[3, 224, 224], use_emb=False):
    if isinstance(net, nn.DataParallel):
        net = net.module

    flop, _ = profile(copy.deepcopy(net), [1] + data_shape, use_emb=use_emb)
    return flop


def measure_net_latency(
        net, l_type="gpu8", fast=True, input_shape=(3, 224, 224), clean=True
):
    if isinstance(net, nn.DataParallel):
        net = net.module

    # remove bn from graph
    rm_bn_from_net(net)

    # return `ms`
    if "gpu" in l_type:
        l_type, batch_size = l_type[:3], int(l_type[3:])
    else:
        batch_size = 1

    data_shape = [batch_size] + list(input_shape)
    if l_type == "cpu":
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
        if get_net_device(net) != torch.device("cpu"):
            if not clean:
                print("move net to cpu for measuring cpu latency")
            net = copy.deepcopy(net).cpu()
    elif l_type == "gpu":
        if fast:
            n_warmup = 5
            n_sample = 10
        else:
            n_warmup = 50
            n_sample = 50
    else:
        raise NotImplementedError
    test_input = torch.zeros(data_shape, device=get_net_device(net))

    measured_latency = {"warmup": [], "sample": []}
    net.eval()
    with torch.no_grad():
        for i in range(n_warmup):
            inner_start_time = time.time()
            net(test_input)
            used_time = (time.time() - inner_start_time) * 1e3  # ms
            measured_latency["warmup"].append(used_time)
            if not clean:
                print("Warmup %d: %.3f" % (i, used_time))
        outer_start_time = time.time()
        for i in range(n_sample):
            net(test_input)
        total_time = (time.time() - outer_start_time) * 1e3  # ms
        measured_latency["sample"].append((total_time, n_sample))
    return total_time / n_sample, measured_latency


def get_net_info(net, input_shape=(3, 224, 224), measure_latency=None, print_info=False):
    net_info = {}
    if isinstance(net, nn.DataParallel):
        net = net.module

    # parameters
    net_info["params"] = count_parameters(net) / 1e6

    # flops
    net_info["flops"] = count_net_flops(net, [1] + list(input_shape)) / 1e6

    # latencies
    latency_types = [] if measure_latency is None else measure_latency.split("#")
    for l_type in latency_types:
        latency, measured_latency = measure_net_latency(
            net, l_type, fast=False, input_shape=input_shape
        )
        net_info["%s latency" % l_type] = {"val": latency, "hist": measured_latency}

    if print_info:
        print(net)
        print("Total training params: %.2fM" % (net_info["params"]))
        print("Total FLOPs: %.2fM" % (net_info["flops"]))
        for l_type in latency_types:
            print(
                "Estimated %s latency: %.3fms"
                % (l_type, net_info["%s latency" % l_type]["val"])
            )

    return net_info
