import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class DropBlock2D(nn.Module):

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)
            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            # place mask on input device
            mask = mask.to(x.device)
            # compute block mask
            block_mask = self._compute_block_mask(mask)
            # apply block mask
            out = x * block_mask[:, None, :, :]
            # scale output
            out = out * block_mask.numel() / block_mask.sum()
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)
        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)

    
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_size=7, drop_prob=0.1):
        super(VGGBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = DropBlock2D(drop_prob=drop_prob, block_size=block_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.drop(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.drop(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class VGGBlock_one(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGBlock_one, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out

class SAUnet(nn.Module):
    def __init__(self, in_ch, num_class):
        super(SAUnet, self).__init__()
        filter = [16, 32, 64, 128]

        self.num_classes = num_class

        self.pool = nn.MaxPool2d(2)

        self.CONV1_1 = VGGBlock(in_ch, filter[0])
        self.CONV2_1 = VGGBlock(filter[0], filter[1])
        self.CONV3_1 = VGGBlock(filter[1], filter[2])
        self.CONV4_1 = VGGBlock_one(filter[2], filter[3])

        self.SAM = SpatialAttentionModule()
        self.CONV4_2 = VGGBlock_one(filter[3], filter[3])
        self.CONV3_2 = VGGBlock(filter[3], filter[2])
        self.CONV2_2 = VGGBlock(filter[2], filter[1])
        self.CONV1_2 = VGGBlock(filter[1], filter[0])
        
        self.sigmod = nn.Sigmoid()

        self.upsample_4_3 = nn.ConvTranspose2d(in_channels=filter[3], out_channels=filter[2], kernel_size=2, stride=2,
                                               padding=0)
        self.upsample_3_2 = nn.ConvTranspose2d(in_channels=filter[2], out_channels=filter[1], kernel_size=2, stride=2,
                                               padding=0)
        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=filter[1], out_channels=filter[0], kernel_size=2, stride=2,
                                               padding=0)
        self.final = nn.Sequential(
            nn.Conv2d(filter[0], self.num_classes, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        
        
    def forward(self, x):
        x_1_1 = self.CONV1_1(x)
        x_2_1 = self.CONV2_1(self.pool(x_1_1))
        x_3_1 = self.CONV3_1(self.pool(x_2_1))
        x_4_1 = self.CONV4_1(self.pool(x_3_1))
        x_4_1 = x_4_1 * self.SAM(x_4_1)
        x_4_2 = self.CONV4_2(x_4_1)
        x_3_2 = self.upsample_4_3(x_4_2)
        x_3_2 = torch.cat([x_3_2, x_3_1], 1)
        x_3_2 = self.CONV3_2(x_3_2)
        x_2_2 = self.upsample_3_2(x_3_2)
        x_2_2 = torch.cat([x_2_2, x_2_1], 1)
        x_2_2 = self.CONV2_2(x_2_2)
        x_1_2 = self.upsample_2_1(x_2_2)
        x_1_2 = torch.cat([x_1_2, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)
        out = self.final(x_1_2)

        return out
