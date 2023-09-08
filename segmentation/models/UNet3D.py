import torch
import torch.nn as nn
from torch.nn import functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        channels = out_channels // 2
        if in_channels > out_channels:
            channels = in_channels // 2

        layers = [
            nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(True),

            nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True)
        ]

        # 构造序列器
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__()
        self.maxpool_to_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpSampling, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + in_channels // 2, out_channels)


    def forward(self, inputs1, inputs2):
        inputs1 = self.up(inputs1)

        outputs = torch.cat((inputs1, inputs2), dim=1)
        outputs = self.conv(outputs)
        return outputs


class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LastConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        conv_out = self.conv(x)
        out = self.Softmax(conv_out)
        return out


class UNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channel=2, bilinear=True, training=True):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear
        self.training = training

        self.inputs = DoubleConv(in_channels, 64)
        self.down_1 = DownSampling(64, 128)
        self.down_2 = DownSampling(128, 256)
        self.down_3 = DownSampling(256, 512)

        self.up_1 = UpSampling(512, 256, self.bilinear)
        self.up_2 = UpSampling(256, 128, self.bilinear)
        self.up_3 = UpSampling(128, 64, self.bilinear)
        self.outputs = LastConv(64, out_channel)

        self.map3 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        self.map2 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )

        self.map1 = nn.Sequential(
            nn.Conv3d(512, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=False),
            nn.Softmax(dim=1)
        )
        self.map4 = LastConv(64, out_channel)

    def forward(self, x):
        # down
        x1 = self.inputs(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)

        # up
        x5 = self.up_1(x4, x3)
        x6 = self.up_2(x5, x2)
        x7 = self.up_3(x6, x1)

        output1 = self.map1(x4)
        output2 = self.map2(x5)
        output3 = self.map3(x6)
        output4 = self.map4(x7)
        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4
