# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule, ModuleList, Sequential
from torch import Tensor
from typing import Optional, Tuple, Union
from mmseg.utils import OptConfigType


class DAPPM(BaseModule):
    """DAPPM module in `DDRNet <https://arxiv.org/abs/2101.06085>`_.

    Args:
        in_channels (int): Input channels.
        branch_channels (int): Branch channels.
        out_channels (int): Output channels.
        num_scales (int): Number of scales.
        kernel_sizes (list[int]): Kernel sizes of each scale.
        strides (list[int]): Strides of each scale.
        paddings (list[int]): Paddings of each scale.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU', inplace=True).
        conv_cfg (dict): Config dict for convolution layer in ConvModule.
            Default: dict(order=('norm', 'act', 'conv'), bias=False).
        upsample_mode (str): Upsample mode. Default: 'bilinear'.
    """

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', momentum=0.1),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(
                     order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear'):
        super().__init__()

        self.num_scales = num_scales
        self.unsample_mode = upsample_mode
        self.in_channels = in_channels
        self.branch_channels = branch_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv_cfg = conv_cfg

        self.scales = ModuleList([
            ConvModule(
                in_channels,
                branch_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **conv_cfg)
        ])
        for i in range(1, num_scales - 1):
            self.scales.append(
                Sequential(*[
                    nn.AvgPool2d(
                        kernel_size=kernel_sizes[i - 1],
                        stride=strides[i - 1],
                        padding=paddings[i - 1]),
                    ConvModule(
                        in_channels,
                        branch_channels,
                        kernel_size=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        **conv_cfg)
                ]))
        self.scales.append(
            Sequential(*[
                nn.AdaptiveAvgPool2d((1, 1)),
                ConvModule(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **conv_cfg)
            ]))
        self.processes = ModuleList()
        for i in range(num_scales - 1):
            self.processes.append(
                ConvModule(
                    branch_channels,
                    branch_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **conv_cfg))

        self.compression = ConvModule(
            branch_channels * num_scales,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

        self.shortcut = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

    def forward(self, inputs: Tensor):
        feats = []
        feats.append(self.scales[0](inputs))

        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs),
                size=inputs.shape[2:],
                mode=self.unsample_mode)
            feats.append(self.processes[i - 1](feat_up + feats[i - 1]))

        return self.compression(torch.cat(feats,
                                          dim=1)) + self.shortcut(inputs)


class PAPPM(DAPPM):
    """PAPPM module in `PIDNet <https://arxiv.org/abs/2206.02066>`_.

    Args:
        in_channels (int): Input channels.
        branch_channels (int): Branch channels.
        out_channels (int): Output channels.
        num_scales (int): Number of scales.
        kernel_sizes (list[int]): Kernel sizes of each scale.
        strides (list[int]): Strides of each scale.
        paddings (list[int]): Paddings of each scale.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.1).
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU', inplace=True).
        conv_cfg (dict): Config dict for convolution layer in ConvModule.
            Default: dict(order=('norm', 'act', 'conv'), bias=False).
        upsample_mode (str): Upsample mode. Default: 'bilinear'.
    """

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', momentum=0.1),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(
                     order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear'):
        super().__init__(in_channels, branch_channels, out_channels,
                         num_scales, kernel_sizes, strides, paddings, norm_cfg,
                         act_cfg, conv_cfg, upsample_mode)

        self.processes = ConvModule(
            self.branch_channels * (self.num_scales - 1),
            self.branch_channels * (self.num_scales - 1),
            kernel_size=3,
            padding=1,
            groups=self.num_scales - 1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            **self.conv_cfg)

    def forward(self, inputs: Tensor):
        x_ = self.scales[0](inputs)
        feats = []
        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs),
                size=inputs.shape[2:],
                mode=self.unsample_mode,
                align_corners=False)
            feats.append(feat_up + x_)
        scale_out = self.processes(torch.cat(feats, dim=1))
        return self.compression(torch.cat([x_, scale_out],
                                          dim=1)) + self.shortcut(inputs)


class RPPM(DAPPM):
    """RPPM.

        Args:
            in_channels (int): Input channels.
            branch_channels (int): Branch channels.
            out_channels (int): Output channels.
            num_scales (int): Number of scales.
            kernel_sizes (list[int]): Kernel sizes of each scale.
            strides (list[int]): Strides of each scale.
            paddings (list[int]): Paddings of each scale.
            norm_cfg (dict): Config dict for normalization layer.
                Default: dict(type='BN', momentum=0.1).
            act_cfg (dict): Config dict for activation layer in ConvModule.
                Default: dict(type='ReLU', inplace=True).
            conv_cfg (dict): Config dict for convolution layer in ConvModule.
                Default: dict(order=('norm', 'act', 'conv'), bias=False).
            upsample_mode (str): Upsample mode. Default: 'bilinear'.
            deploy (bool): Whether in deploy mode. Default: False.
        """

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', requires_grad=True),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(
                     order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear',
                 deploy: bool = False):
        super().__init__(in_channels, branch_channels, out_channels,
                         num_scales, kernel_sizes, strides, paddings, norm_cfg,
                         act_cfg, conv_cfg, upsample_mode)

        self.deploy = deploy

        self.processes = RepParallel(
            self.branch_channels * (self.num_scales - 1),
            self.branch_channels * (self.num_scales - 1),
            kernel_size=3,
            padding=1,
            groups=self.num_scales - 1,
            norm_cfg=self.norm_cfg,
            deploy=self.deploy,
        )

    def forward(self, inputs: Tensor):
        x_ = self.scales[0](inputs)
        feats = []
        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs),
                size=inputs.shape[2:],
                mode=self.unsample_mode,
                align_corners=False)
            feats.append(feat_up + x_)
        scale_out = self.processes(torch.cat(feats, dim=1))
        return self.compression(torch.cat([x_, scale_out],
                                          dim=1)) + self.shortcut(inputs)

    def switch_to_deploy(self):
        for m in self.modules():
            if isinstance(m, RepParallel):
                m.switch_to_deploy()


class RepParallel(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 1,
                 groups: Optional[int] = 1,
                 padding_mode: Optional[str] = 'zeros',
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True, momentum=0.03, eps=0.001),
                 deploy: bool = False):
        super().__init__()

        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        self.norm = build_norm_layer(norm_cfg, in_channels)[1]
        self.relu = nn.ReLU(inplace=True)

        if deploy:
            self.reparam_parallel = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=True,
                padding_mode=padding_mode)

        else:
            self.parallel1 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.parallel2 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)

    def forward(self, inputs: Tensor) -> Tensor:

        inputs = self.relu(self.norm(inputs))

        if hasattr(self, 'reparam_parallel'):
            return self.reparam_parallel(inputs)

        return self.parallel1(inputs) + self.parallel2(inputs)

    def get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias in a differentiable way.

        Returns:
            tuple: Equivalent kernel and bias
        """
        kernel1, bias1 = self._fuse_bn_tensor(self.parallel1)
        kernel2, bias2 = self._fuse_bn_tensor(self.parallel2)

        return kernel1 + kernel2, bias1 + bias2

    def _fuse_bn_tensor(self, conv: nn.Module) -> Tuple[np.ndarray, Tensor]:
        """Derives the equivalent kernel and bias of a specific conv layer.

        Args:
            conv (nn.Module): The layer that needs to be equivalently
                transformed, which can be nn.Sequential or nn.Batchnorm2d

        Returns:
            tuple: Equivalent kernel and bias
        """
        kernel = conv.conv.weight
        running_mean = conv.bn.running_mean
        running_var = conv.bn.running_var
        gamma = conv.bn.weight
        beta = conv.bn.bias
        eps = conv.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """Switch to deploy mode."""
        if hasattr(self, 'reparam_parallel'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_parallel = nn.Conv2d(
            in_channels=self.parallel1.conv.in_channels,
            out_channels=self.parallel1.conv.out_channels,
            kernel_size=self.parallel1.conv.kernel_size,
            stride=self.parallel1.conv.stride,
            padding=self.parallel1.conv.padding,
            groups=self.parallel1.conv.groups,
            bias=True)
        self.reparam_parallel.weight.data = kernel
        self.reparam_parallel.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('parallel1')
        self.__delattr__('parallel2')
        self.deploy = True
