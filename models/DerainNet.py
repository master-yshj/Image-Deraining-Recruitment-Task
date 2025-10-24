import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use1x1conv=False):
        super(ResidualBlock, self).__init__()
        self.use1x1conv=use1x1conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if use1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.LeakyReLU(y)
        y = self.conv2(y)
        if self.use1x1conv:
            x = self.conv3(x)
        y = y + x
        return y


class DerainNet(nn.Module):
    def __init__(self):
        super(DerainNet, self).__init__()
        self.in_channels = 3
        self.mid_channels = 32
        self.out_channels = 3
        self.conv1 = nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(self.mid_channels * 3, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)#1/2用一次，1/4用两次

        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)

        self.block = ResidualBlock(self.mid_channels, self.mid_channels, use1x1conv=False)

    def upsample(self, x, scale_factor):
        out = F.interpolate(
            x,
            scale_factor=scale_factor,  # 放大倍数
            mode='bilinear',  # 插值方式
            align_corners=True  # 对齐角落像素（避免边缘失真）
        )
        out = self.conv2(out)
        out = self.LeakyReLU(out)
        out = self.conv2(out)
        return out

    def forward(self, input):
        x = self.conv1(input)
        x = self.LeakyReLU(x)
        x1 = x
        x2 = self.maxpool(x1)
        x3 = self.maxpool(x2)

        """分支一"""
        for i in range(8):
            x1 = self.block(x1)
        """分支二"""
        for i in range(8):
            x2 = self.block(x2)
        x2 = self.upsample(x2, scale_factor=2)
        """分支三"""
        for i in range(8):
            x3 = self.block(x3)
        x3 = self.upsample(x3, scale_factor=4)
        """合并三个图像"""
        out = torch.cat((x1, x2, x3), dim = 1)
        out = self.conv3(out)
        out = self.conv4(out)
        final_out = input-out
        return final_out

