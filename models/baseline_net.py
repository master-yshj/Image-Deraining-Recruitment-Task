import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, stride=1,use_1x1conv=False):
        super(ResidualBlock, self).__init__()
        self.use_1x1conv=use_1x1conv
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False)

        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)


    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.use_1x1conv:
            x = self.conv3(x)
        y += x
        y = self.relu(y)
        return y

class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.block1 = ResidualBlock(in_channel=3, mid_channel=64, out_channel=64, use_1x1conv=True)
        self.block2 = ResidualBlock(in_channel=64, mid_channel=64, out_channel=64, use_1x1conv=False)
        self.block3 = ResidualBlock(in_channel=64, mid_channel=64, out_channel=64, use_1x1conv=False)
        self.block4 = ResidualBlock(in_channel=64, mid_channel=64, out_channel=3, use_1x1conv=True)

    def forward(self, x):
        r = self.block1(x)
        r = self.block2(r)
        r = self.block3(r)
        r = self.block4(r)
        y = x-r
        return y


