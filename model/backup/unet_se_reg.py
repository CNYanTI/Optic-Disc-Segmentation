import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#   https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html
class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.leaky_relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


def _conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class ContinusParalleConv_SEBlock(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', drop_prob=0.1, block_size=7, deploy=False):
        super(ContinusParalleConv_SEBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = deploy
        self.groups = groups

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.LeakyReLU()

        self.post_se = SEBlock(input_channels=out_channels, internal_neurons=out_channels // 16)
        
        self.post_drop = nn.Dropout(0.3)
        
        # self.post_drop = DropBlock2D(drop_prob=drop_prob, block_size=block_size)

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

            self._rbr_reparam = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
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

    def forward(self, x):# current: conv-bn-relu-conv-bn-relu-se
        if hasattr(self, 'rbr_reparam'):
            # 使用结构重参数化了的卷积核，每层两次卷积
            x = self.nonlinearity(self.rbr_reparam(x))
            return self.post_se(self.nonlinearity(self._rbr_reparam(x)))

        # 第一步卷积
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)
        x = self.rbr_dense(x) + self.rbr_1x1(x) + id_out    # conv+bn
        x = self.nonlinearity(x)                            # relu
        # x = self.rbr_dense(x)
        # x = self.nonlinearity(x)

        # 第二步卷积
        if self._rbr_identity is None:
            _id_out = 0
        else:
            _id_out = self._rbr_identity(x)
        out = self._rbr_dense(x) + self._rbr_1x1(x) + _id_out   # conv+bn
        out = self.post_drop(out)
        out = self.post_se(self.nonlinearity(out))                            # SEBlock+relu
        # out = self._rbr_dense(x)
        # out = self.post_drop(out)
        # out = self.nonlinearity(out)

        return out

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        _kernel3x3, _bias3x3 = self._fuse_bn_tensor(self._rbr_dense)
        _kernel1x1, _bias1x1 = self._fuse_bn_tensor(self._rbr_1x1)
        _kernelid, _biasid = self._fuse_bn_tensor(self._rbr_identity)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid, _kernel3x3 + self._pad_1x1_to_3x3_tensor(_kernel1x1) + _kernelid, _bias3x3 + _bias1x1 + _biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        # 将1x1的卷积核用 0 填充成 3x3 的卷积核
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        # 融合 CONV 和 BN
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.out_channels // self.groups
                kernel_value = np.zeros((self.out_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.out_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias, _kernel, _bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self._rbr_reparam = nn.Conv2d(in_channels=self._rbr_dense.conv.in_channels, out_channels=self._rbr_dense.conv.out_channels,
                                     kernel_size=self._rbr_dense.conv.kernel_size, stride=self._rbr_dense.conv.stride,
                                     padding=self._rbr_dense.conv.padding, dilation=self._rbr_dense.conv.dilation, groups=self._rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self._rbr_reparam.weight.data = _kernel
        self._rbr_reparam.bias.data = _bias
        self.__delattr__('rbr_dense')
        self.__delattr__('_rbr_dense')
        self.__delattr__('rbr_1x1')
        self.__delattr__('_rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, '_rbr_identity'):
            self.__delattr__('_rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


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
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(in_channel, in_channel // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


class Unet_SE_REG_Block(nn.Module):
    def __init__(self, in_ch, num_classes, deep_supervision=False, deploy=False):
        super(Unet_SE_REG_Block, self).__init__()    # 继承类
        fliter = [64, 128, 256, 512, 1024]
        # fliter = [16, 32, 64, 128, 256]
        self.num_classes = num_classes          # 分类数目

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
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool_0(x_0_0))
        x_2_0 = self.stage_2(self.pool_1(x_1_0))
        x_3_0 = self.stage_3(self.pool_2(x_2_0))
        x_4_0 = self.stage_4(self.pool_3(x_3_0))
        
        x_3_1 = self.upsample_3_1(x_4_0, x_3_0)
        x_3_1 = self.CONV3_1(x_3_1)

        x_2_2 = self.upsample_2_2(x_3_1, x_2_0)
        x_2_2 = self.CONV2_2(x_2_2)

        x_1_3 = self.upsample_1_3(x_2_2, x_1_0)
        x_1_3 = self.CONV1_3(x_1_3)

        x_0_4 = self.upsample_0_4(x_1_3, x_0_0)
        x_0_4 = self.CONV0_4(x_0_4)

        return self.final_super_0_4(x_0_4)

    # 训练的时候不用结构重参数化，在成品最后一步之前使用结构重参数化，在主进程里调用
    def switch_net_to_deploy(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()