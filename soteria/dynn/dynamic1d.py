# _*_ coding: utf-8 _*_
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from soteria.utils import (
    sub_filter_start_end,
    get_same_padding,
    weight_standardization,
    get_net_device,
    copy_bn)

__all__ = [
    "SeparableConv1d",
    "DynamicConv1d_DC",
    "DynamicGroupConv1d",
    "BatchNorm1d",
    "Linear",
    "Dynamic_Module",
]


def Dynamic2Static(name):
    synamic2static = {
        SeparableConv1d.__name__: nn.Conv1d,
        DynamicConv1d_DC.__name__: nn.Conv1d,
        DynamicGroupConv1d.__name__: nn.Conv1d,
        BatchNorm1d.__name__: nn.BatchNorm1d,
    }
    return synamic2static[name]


class Dynamic_Module(nn.Module):
    def forward(self, x):
        raise NotImplementedError

    def set_active(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

    @staticmethod
    def get_active_module(config):
        raise NotImplementedError


class SeparableConv1d(Dynamic_Module):
    KERNEL_TRANSFORM_MODE = 1  

    def __init__(self, max_in_channels, kernel_size_list, stride=1, dilation=1, WS_EPS=None):
        super(SeparableConv1d, self).__init__()

        self.max_in_channels = max_in_channels
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation
        self.WS_EPS = WS_EPS

        self.conv = nn.Conv1d(
            self.max_in_channels,
            self.max_in_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=self.max_in_channels,
            bias=False,
        )

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  

        if self.KERNEL_TRANSFORM_MODE is not None:
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = "%dto%d" % (ks_larger, ks_small)
                scale_params["%s_matrix" % param_name] = Parameter(
                    torch.eye(ks_small)
                )
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)
        self.current_in_channel = None

        self.config = {
            "active_kernel_size": self.active_kernel_size,
        }
        self.cur_output_length = 0

    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[
                           :out_channel, :in_channel, :
                           ]  
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter,
                    self.__getattr__("%dto%d_matrix" % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks
                )
                start_filter = _input_filter
            filters = start_filter

        filters = (
            weight_standardization(filters, self.WS_EPS)
            if self.WS_EPS is not None
            else filters
        )

        return filters

    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)
        self.cur_output_length = x.shape[-1]
        self.current_in_channel = in_channel
        filters = self.get_active_filter(in_channel, kernel_size).contiguous()
        padding = get_same_padding(kernel_size)
        y = F.conv1d(x, filters, None, self.stride, padding, self.dilation, in_channel)
        return y

    def __repr__(self):
        return "SeparableConv1d(%d, %d, kernel=(%d,), stride=%d, dilation=%d)" \
               % (self.max_in_channels,
                  self.max_in_channels,
                  max(self.kernel_size_list),
                  self.stride,
                  self.dilation
                  )

    def get_active_module(self, preserve_weight=True):
        active_module = nn.Conv1d(self.current_in_channel,
                                  self.current_in_channel,
                                  self.active_kernel_size,
                                  stride=self.stride,
                                  groups=self.current_in_channel,
                                  padding=get_same_padding(self.active_kernel_size),
                                  bias=False)
        meta = {"x": [1, self.current_in_channel, 0],
                "y": [1, self.current_in_channel, self.cur_output_length]}
        active_module.meta_info = meta
        active_module = active_module.to(get_net_device(self))
        if not preserve_weight:
            return active_module

        active_module.weight.data.copy_(
            self.get_active_filter(
                self.current_in_channel, self.active_kernel_size
            ).data
        )
        return active_module


class DynamicConv1d_DC(Dynamic_Module):
    def __init__(
            self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1, WS_EPS=None, padding=None
    ):
        super(DynamicConv1d_DC, self).__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.WS_EPS = WS_EPS
        self.padding = padding

        self.conv = nn.Conv1d(
            self.max_in_channels,
            self.max_out_channels,
            self.kernel_size,
            stride=self.stride,
            bias=False,
        )

        self.active_out_channel = self.max_out_channels
        self.current_in_channel = None

        self.config = {
                "active_out_channel": self.active_out_channel,
            }
        self.cur_output_length = 0

    def get_active_filter(self, out_channel, in_channel):
        filters = self.conv.weight[:out_channel, :in_channel, :]
        filters = (
            weight_standardization(filters, self.WS_EPS)
            if self.WS_EPS is not None
            else filters
        )
        return filters

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        self.cur_output_length = x.shape[-1]
        self.current_in_channel = in_channel
        filters = self.get_active_filter(out_channel, in_channel).contiguous()
        if self.padding is None:
            padding = get_same_padding(self.kernel_size)
        else:
            padding = self.padding
        y = F.conv1d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


    def __repr__(self):
        return "DynamicConv1d_DC(%d, %d, kernel=(%d,), stride=%d, dilation=%d)" \
               % (self.max_in_channels,
                  self.max_out_channels,
                  self.kernel_size,
                  self.stride,
                  self.dilation
                  )

    def get_active_module(self, preserve_weight=True):
        active_module = nn.Conv1d(self.current_in_channel,
                                  self.active_out_channel,
                                  self.kernel_size,
                                  stride=self.stride,
                                  padding=self.padding if self.padding is not None
                                  else get_same_padding(self.kernel_size),
                                  bias=False)
        meta = {"x": [1, self.current_in_channel, 0],
                "y": [1, self.active_out_channel, self.cur_output_length]}
        active_module.meta_info = meta
        active_module = active_module.to(get_net_device(self))
        if not preserve_weight:
            return active_module

        active_module.weight.data.copy_(
            self.get_active_filter(
                self.active_out_channel, self.current_in_channel
            ).data
        )
        return active_module


class DynamicGroupConv1d(Dynamic_Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size_list,
            groups_list,
            stride=1,
            dilation=1,
            WS_EPS=None
    ):
        super(DynamicGroupConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_list = kernel_size_list
        self.groups_list = groups_list
        self.stride = stride
        self.dilation = dilation
        self.WS_EPS = WS_EPS

        self.conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=min(self.groups_list),
            bias=False,
        )
        self.cur_output_length = 0
        self.active_kernel_size = max(self.kernel_size_list)
        self.active_groups = min(self.groups_list)

        self.config = {
                "active_kernel_size": self.active_kernel_size,
                "active_groups": self.active_groups
            }

    def get_active_filter(self, kernel_size, groups):
        start, end = sub_filter_start_end(max(self.kernel_size_list), kernel_size)
        filters = self.conv.weight[:, :, start:end]

        sub_filters = torch.chunk(filters, groups, dim=0)
        sub_in_channels = self.in_channels // groups
        sub_ratio = filters.size(1) // sub_in_channels

        filter_crops = []
        for i, sub_filter in enumerate(sub_filters):
            part_id = i % sub_ratio
            start = part_id * sub_in_channels
            filter_crops.append(sub_filter[:, start: start + sub_in_channels, :])
        filters = torch.cat(filter_crops, dim=0)
        filters = (
            weight_standardization(filters, self.WS_EPS)
            if self.WS_EPS is not None
            else filters
        )
        return filters

    def forward(self, x, kernel_size=None, groups=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        if groups is None:
            groups = self.active_groups
        self.cur_output_length = x.shape[-1]
        filters = self.get_active_filter(kernel_size, groups).contiguous()
        padding = get_same_padding(kernel_size)
        y = F.conv1d(
            x,
            filters,
            None,
            self.stride,
            padding,
            self.dilation,
            groups,
        )
        return y

    def __repr__(self):
        return "DynamicGroupConv1d(%d, %d, kernel=(%d,), stride=%d, dilation=%d, group=%d)" \
               % (self.in_channels,
                  self.out_channels,
                  max(self.kernel_size_list),
                  self.stride,
                  self.dilation,
                  max(self.groups_list)
                  )

    def get_active_module(self, preserve_weight=True):
        active_module = nn.Conv1d(self.in_channels,
                                  self.out_channels,
                                  self.active_kernel_size,
                                  stride=self.stride,
                                  padding=get_same_padding(),
                                  groups=self.active_groups,
                                  bias=False)
        meta = {"x": [1, self.in_channels, 0],
                "y": [1, self.out_channels, self.cur_output_length]}
        active_module.meta_info = meta
        active_module = active_module.to(get_net_device(self))
        if not preserve_weight:
            return active_module

        active_module.weight.data.copy_(
            self.get_active_filter(
                self.active_kernel_size, self.active_groups
            ).data
        )
        return active_module


class BatchNorm1d(Dynamic_Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, max_feature_dim):
        super(BatchNorm1d, self).__init__()
        self.input_record = None
        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm1d(self.max_feature_dim)
        self.current_feature_dim = self.max_feature_dim

    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm1d, feature_dim):
        if bn.num_features == feature_dim or BatchNorm1d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                exponential_average_factor,
                bn.eps,
            )

    def forward(self, x):
        self.input_record = x.shape
        feature_dim = x.size(1)
        self.current_feature_dim = feature_dim
        y = self.bn_forward(x, self.bn, feature_dim)
        return y

    def __repr__(self):
        return self.bn.__repr__()

    def get_active_module(self):
        active_module = nn.BatchNorm1d(self.current_feature_dim)
        active_module.meta_info = dict(x=self.input_record, y=[0])
        copy_bn(active_module, self.bn)



        return active_module



class Linear(Dynamic_Module):
    def __init__(self, max_in_features, max_out_features, bias=True):
        super(Linear, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias

        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)

        self.active_out_features = self.max_out_features
        self.current_in_features = None

        self.config = {
                "active_out_features": self.active_out_features
            }

    def get_active_weight(self, out_features, in_features):
        return self.linear.weight[:out_features, :in_features]

    def get_active_bias(self, out_features):
        return self.linear.bias[:out_features] if self.bias else None

    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(1)
        self.current_in_features = in_features
        weight = self.get_active_weight(out_features, in_features).contiguous()
        bias = self.get_active_bias(out_features)
        y = F.linear(x, weight, bias)
        return y

    def __repr__(self):
        return "Linear(%d, %d, bias=%s)" % (self.max_in_features, self.max_out_features, str(self.bias))

    def get_active_module(self, preserve_weight=True):
        active_module = nn.Linear(self.current_in_features, self.active_out_features, self.bias)
        # active_module.__repr__ = self.__repr__
        meta = {"x": self.current_in_features,
                "y": self.active_out_features}
        active_module.meta_info = meta
        active_module = active_module.to(get_net_device(self))
        if not preserve_weight:
            return active_module
        active_module.weight.data.copy_(self.get_active_weight(self.active_out_features, self.current_in_features).data)
        if self.bias:
            active_module.bias.data.copy_(self.get_active_bias(self.active_out_features).data)
        return active_module
