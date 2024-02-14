import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd
import cv2 as cv2

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, groups=groups, bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class ContinusParalleConv_SEBlock(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 padding_mode='zeros', drop_prob=0.1, block_size=7, deploy=False):
        super(ContinusParalleConv_SEBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = deploy
        self.groups = groups

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2
        
        self.nonlinearity = nn.LeakyReLU()
        
        self.post_drop = nn.Dropout(0.3)
        
        # 第一个卷积
        self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, groups=groups)
        self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                               padding=padding_11, groups=groups)

        # 第二个卷积
        self._rbr_identity = nn.BatchNorm2d(num_features=out_channels) if stride == 1 else None
        self._rbr_dense = conv_bn(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, groups=groups)
        self._rbr_1x1 = conv_bn(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                padding=padding_11, groups=groups)

    def forward(self, x):

        x = self.rbr_dense(x) + self.rbr_1x1(x)
        x = self.nonlinearity(x)

        out = self._rbr_dense(x) + self._rbr_1x1(x)
        out = self.post_drop(out)
        out = self.nonlinearity(out)

        return out
        
# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


class UpSampling(nn.Module):

    def __init__(self, in_channel):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(in_channel, in_channel // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        return torch.cat((x, r), 1)
    
class BackBoneRes(nn.Module):
    def __init__(self, in_ch, num_classes, deep_supervision=False, deploy=False):
        super(BackBoneRes, self).__init__()  # 继承类
        fliter = [32, 64, 128, 256, 512]
        # fliter = [16, 32, 64, 128, 256]
        self.num_classes = num_classes  # 分类数目

        self.CONV3_1 = ContinusParalleConv_SEBlock(fliter[3] * 2, fliter[3])
        self.CONV2_2 = ContinusParalleConv_SEBlock(fliter[2] * 2, fliter[2])
        self.CONV1_3 = ContinusParalleConv_SEBlock(fliter[1] * 2, fliter[1])
        self.CONV0_4 = ContinusParalleConv_SEBlock(fliter[0] * 2, fliter[0])

        self.stage_0 = ContinusParalleConv_SEBlock(in_ch, fliter[0])
        self.stage_1 = ContinusParalleConv_SEBlock(fliter[0], fliter[1])
        self.stage_2 = ContinusParalleConv_SEBlock(fliter[1], fliter[2])
        self.stage_3 = ContinusParalleConv_SEBlock(fliter[2], fliter[3])
        self.stage_4 = ContinusParalleConv_SEBlock(fliter[3], fliter[4])

        self.pool_0 = DownSampling(fliter[0])
        self.pool_1 = DownSampling(fliter[1])
        self.pool_2 = DownSampling(fliter[2])
        self.pool_3 = DownSampling(fliter[3])

        self.upsample_3_1 = UpSampling(fliter[4])
        self.upsample_2_2 = UpSampling(fliter[3])
        self.upsample_1_3 = UpSampling(fliter[2])
        self.upsample_0_4 = UpSampling(fliter[1])

        self.final_super_0_4 = nn.Sequential(
            nn.Conv2d(fliter[0], self.num_classes, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_size = x.size()

        # main
        x_0_0 = self.stage_0(x)

        x_1_0 = self.stage_1(self.pool_0(x_0_0))
        x_2_0 = self.stage_2(self.pool_1(x_1_0))
        x_3_0 = self.stage_3(self.pool_2(x_2_0))
        x_4_0 = self.stage_4(self.pool_3(x_3_0))

        _x_3_1 = self.upsample_3_1(x_4_0, x_3_0)
        x_3_1 = self.CONV3_1(_x_3_1)

        _x_2_2 = self.upsample_2_2(x_3_1, x_2_0)
        x_2_2 = self.CONV2_2(_x_2_2)

        _x_1_3 = self.upsample_1_3(x_2_2, x_1_0)
        x_1_3 = self.CONV1_3(_x_1_3)

        _x_0_4 = self.upsample_0_4(x_1_3, x_0_0)
        x_0_4 = self.CONV0_4(_x_0_4)
        x_0_4 = self.final_super_0_4(x_0_4)

        return x_0_4