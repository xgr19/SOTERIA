# _*_ coding: utf-8 _*_
import torch
import torch.nn as nn
import soteria.dynn as dynn
from soteria.utils import *
from collections import OrderedDict

CHANNEL_DIVISIBLE = 8


class CrossFeature(dynn.Dynamic_Layer):
    def __init__(self, in_channel, seq_len, target_len):
        super(CrossFeature, self).__init__()
        self.channels_fuse = nn.Sequential(nn.Conv1d(in_channel, in_channel, kernel_size=1),
                                           nn.BatchNorm1d(in_channel),
                                           nn.ReLU(inplace=True))
        self.proj = nn.Linear(seq_len, target_len)
        self.repr = "CrossFeature-%d-%d-%d" % (in_channel, seq_len, target_len)

    def forward(self, x):
        # x (bs, in_c, seq_len)
        x = self.channels_fuse(x)
        return self.proj(x)

    def __repr__(self):
        return self.repr

class FeatureExtract(dynn.Dynamic_Layer):
    def __init__(self,
                 seq_len=23,
                 max_embed_dim=16,
                 out_fea=256):
        super(FeatureExtract, self).__init__()
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(seq_len, out_fea),
                      nn.BatchNorm1d(out_fea),
                      nn.ReLU(inplace=True)) for _ in range(max_embed_dim)])
        self.repr = "FeatureExtract-%d-%d-%d" % (seq_len, max_embed_dim, out_fea)

    def forward(self, x):
        out = []
        for layer in self.layers:
            out.append(layer(x).unsqueeze(1))
        return torch.concat(out, dim=1)

    def __repr__(self):
        return self.repr



class PatchEmbed(dynn.Dynamic_Layer):
    def __init__(self,
                 seq_len=23,
                 patch_size=23,
                 in_channels=1,
                 max_embed_dim=8):
        super(PatchEmbed, self).__init__()
        self.proj = dynn.DynamicConv1d_DC(in_channels,
                                          max_embed_dim,
                                          kernel_size=patch_size,
                                          stride=1,
                                          padding=patch_size//2)
        self.norm = dynn.BatchNorm1d(max_embed_dim)
        self.active_embed_dim = max_embed_dim
        self.repr = "PathEmb-%d-%d-%d-%d" % (seq_len, patch_size, in_channels, max_embed_dim)

    def __repr__(self):
        return self.repr

    def forward(self, x):
        self.proj.active_out_channel = self.active_embed_dim
        x = self.proj(x.unsqueeze(1)) 
        x = self.norm(x)
        return x

    @property
    def add_config(self):
        cover = True 
        config = {
            "active_embed_dim": self.active_embed_dim,
        }
        return config, cover


class MBConvLayer(nn.Module):
    CHANNEL_DIVISIBLE = 8

    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio=1,
                 kernel_size=7,
                 stride=2,
                 act_func="h_swish"
                 ):
        super(MBConvLayer, self).__init__()
        self.in_channels = in_channels
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.middle_channel = make_divisible(
            round(self.in_channels * self.expand_ratio),
            CHANNEL_DIVISIBLE,
        )
        self.inverted_bottleneck = nn.Sequential(nn.Conv1d(self.in_channels, self.middle_channel, 1),
                                                 nn.BatchNorm1d(self.middle_channel),
                                                 build_activation(act_func))

        self.depth_conv = nn.Sequential(nn.Conv1d(self.middle_channel,
                                                  self.middle_channel,
                                                  self.kernel_size,
                                                  self.stride,
                                                  padding=get_same_padding(self.kernel_size),
                                                  groups=self.middle_channel),
                                        nn.BatchNorm1d(self.middle_channel),
                                        build_activation(act_func))
        self.point_linear = nn.Sequential(nn.Conv1d(self.middle_channel, self.out_channels, 1),
                                          nn.BatchNorm1d(self.out_channels))

    def forward(self, x):
        h = self.inverted_bottleneck(x)
        h = self.depth_conv(h)
        h = self.point_linear(h)
        return h


class Dynamic_MBConvLayer(dynn.Dynamic_Layer):
    

    def __init__(self,
                 max_in_channels,
                 expand_ratio_list,
                 kernel_size_list,
                 stride=1,
                 act_func="h_swish",
                 active=False
                 ):
        super(Dynamic_MBConvLayer, self).__init__()
        self.max_in_channels = max_in_channels
        self.expand_ratio_list = int2list(expand_ratio_list)
        self.max_expand_ratio = max(self.expand_ratio_list)
        self.kernel_size_list = int2list(kernel_size_list)
        self.stride = stride
        self.act_func = act_func
        self.max_out_channels = self.max_in_channels if stride == 1 else self.max_in_channels * 2
        self.max_middle_channels = make_divisible(
            round(self.max_in_channels * self.max_expand_ratio),
            CHANNEL_DIVISIBLE,
        )
       
        self.inverted_bottleneck = dynn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        dynn.DynamicConv1d_DC(
                            self.max_in_channels, self.max_middle_channels
                        ),
                    ),
                    ("bn", dynn.BatchNorm1d(self.max_middle_channels)),
                    ("act", build_activation(self.act_func)),
                ]
            )
        )
        self.depth_conv = dynn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        dynn.SeparableConv1d(
                            self.max_middle_channels, self.kernel_size_list, self.stride
                        ),
                    ),
                    ("bn", dynn.BatchNorm1d(self.max_middle_channels)),
                    ("act", build_activation(self.act_func)),
                ]
            )
        )

        self.point_linear = dynn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        dynn.DynamicConv1d_DC(self.max_middle_channels, self.max_out_channels),
                    ),
                    ("bn", dynn.BatchNorm1d(self.max_out_channels)),
                ]
            )
        )
        
        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = self.max_expand_ratio
        self.repr = "Dynamic_MBConvLayer-%d-%d-%d-%d-%s" % (
            max_in_channels, self.active_kernel_size, self.active_expand_ratio, stride, act_func)
        if active:
            x = torch.ones([1, max_in_channels, 256])
            _ = self.forward(x)
            self.get_active_module()

    def forward(self, x):
     
        in_channel = x.size(1)
        # self.inverted_bottleneck.conv.active_out_channel = make_divisible(
        #         round(in_channel * self.active_expand_ratio),
        #         MBConvLayer.CHANNEL_DIVISIBLE,
        #     )
        self.inverted_bottleneck.conv.active_out_channel = in_channel * self.active_expand_ratio

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.point_linear.conv.active_out_channel = in_channel if self.stride == 1 else in_channel * 2

        x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    def __repr__(self):
        return self.repr

    @property
    def add_config(self):
        cover = True 
        config = {
            "active_kernel_size": self.active_kernel_size,
            "active_expand_ratio": self.active_expand_ratio,
        }
        return config, cover

    def re_organize_middle_weights(self):
        importance = torch.sum(
            torch.abs(self.point_linear.conv.conv.weight.data), dim=(0, 2)
        )

        sorted_importance, sorted_idx = torch.sort(importance, dim=0, descending=True)
        self.point_linear.conv.conv.weight.data = torch.index_select(
            self.point_linear.conv.conv.weight.data, 1, sorted_idx
        )

        adjust_bn_according_to_idx(self.depth_conv.bn.bn, sorted_idx)
        self.depth_conv.conv.conv.weight.data = torch.index_select(
            self.depth_conv.conv.conv.weight.data, 0, sorted_idx
        )

        if self.inverted_bottleneck is not None:
            adjust_bn_according_to_idx(self.inverted_bottleneck.bn.bn, sorted_idx)
            self.inverted_bottleneck.conv.conv.weight.data = torch.index_select(
                self.inverted_bottleneck.conv.conv.weight.data, 0, sorted_idx
            )
            return None
        else:
            return sorted_idx



class Dynamic_MobileNet1d(dynn.Dynamic_Layer):
    def __init__(self, cfg, id=0):
        super(Dynamic_MobileNet1d, self).__init__()
        self.kernel_size_list = cfg.kernel_size_list
        self.expand_ratio_list = cfg.expand_ratio_list
        self.depth_list = cfg.depth_list
        self.n_classes = cfg.num_classes
        self.id = id

        base_width = [32, 64, 128, 256]
        self.width_list = [round(make_divisible(width * cfg.width_mult, CHANNEL_DIVISIBLE)) for width in base_width]
        
        if cfg.get("use_emb", False):
            from .mobilenetv3_dga import Embeddings
            self.PathEmb = Embeddings(seq_len=cfg.num_emb,
                                      max_embed_dim=self.width_list[0])
        else:
            self.PathEmb = PatchEmbed()

        module_list0 = [Dynamic_MBConvLayer(
            self.width_list[0] if i == 0 else self.width_list[1],
            self.expand_ratio_list,
            self.kernel_size_list,
            stride=2 if i == 0 else 1,
            act_func=cfg.act_func_list[0]
        ) for i in range(max(self.depth_list))]
        self.stage0 = dynn.ModuleList(module_list0)

        module_list1 = [Dynamic_MBConvLayer(
            self.width_list[1] if i == 0 else self.width_list[2],
            self.expand_ratio_list,
            self.kernel_size_list,
            stride=2 if i == 0 else 1,
            act_func=cfg.act_func_list[1]
        ) for i in range(max(self.depth_list))]
        self.stage1 = dynn.ModuleList(module_list1)

      
        self.Stem2 = Dynamic_MBConvLayer(
            self.width_list[2],
            self.expand_ratio_list,
            self.kernel_size_list,
            stride=2,
            act_func=cfg.act_func_list[2])

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = dynn.Linear(self.width_list[3], self.n_classes)

        self.active_stem0_depth = max(self.depth_list)
        self.active_stem1_depth = max(self.depth_list)

      
        self.fc.config = {}
        self.fc.meta_info = dict(x=[0], y=[0])


    def forward(self, x):
        h = self.PathEmb(x)
        for i in range(self.active_stem0_depth):
            h = self.stage0[i](h)
        for i in range(self.active_stem1_depth):
            h = self.stage1[i](h)
        h = self.Stem2(h)

        self.avgpool.meta_info = dict(x=h.shape)
        h = self.avgpool(h)
        self.avgpool.meta_info.update(y=h.shape)

        h = h.view(h.size(0), -1).contiguous()
        y = self.fc(h)
        return y

    def __repr__(self):
        return "Dynamic_MobileNet1d" + str(self.id)

    def get_active_module(self):
        max_depth = max(self.depth_list)
        # stage0
        for i in range(max_depth - self.active_stem0_depth):
            self.stage0.__delitem__(-1)
        # stage1
        for i in range(max_depth - self.active_stem1_depth):
            self.stage1.__delitem__(-1)

        for name, child in self.named_children():
            if isinstance(child, dynn.Dynamic_Module):
                active_module = self.__getattr__(name).get_active_module()
                self.__setattr__(name, active_module)
            elif isinstance(child, dynn.Dynamic_Layer):
                child.get_active_module()

    @property
    def add_config(self):
        cover = False  
        config = {
            "active_stem0_depth": self.active_stem0_depth,
            "active_stem1_depth": self.active_stem1_depth,
        }
        return config, cover

    def re_organize_middle_weights(self):
        for conv in self.stage0:
            conv.re_organize_middle_weights()
        for conv in self.stage1:
            conv.re_organize_middle_weights()
        self.Stem2.re_organize_middle_weights()
