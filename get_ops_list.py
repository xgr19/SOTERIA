# _*_ coding: utf-8 _*_
from mmt.converter import register
import torch.nn as nn
from soteria.dynn import Linear
from soteria.model import PatchEmbed,Dynamic_MBConvLayer


fp = "ops_list"
reg = lambda ops, **kwargs: register(ops, fp, **kwargs)

reg(PatchEmbed,
    max_embed_dim=[8],
    patch_size=[23],
    in_channels=[1],
    input_shape=[[1, 23]]
    )

reg(Dynamic_MBConvLayer,
    max_in_channels=[8, 16, 32],
    active=[True],
    kernel_size_list=[1, 3, 5, 7],
    expand_ratio_list=[3, 4, 6],
    stride=[1, 2],
    act_func=["relu6", "h_swish"],
    input_shape=[[1, 8, 23], [1, 16, 12], [1, 32, 6]]
    )

reg(nn.AdaptiveAvgPool1d,
    output_size=[1],
    input_shape=[[1, 64, 3]]
    )

reg(Linear,
    max_in_features=[64],
    max_out_features=[6],
    bias=[True],
    input_shape=[[1, 64]],
    )

reg(Linear,
    max_in_features=[64],
    max_out_features=[10],
    bias=[True],
    input_shape=[[1, 64]],
    )