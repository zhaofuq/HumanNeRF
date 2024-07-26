import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, cfg, n_channels, channels = [32, 64, 128, 256, 512], bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, channels[0])
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(channels[3], channels[4] // factor)
        self.up1 = Up(channels[4], channels[3] // factor, bilinear)
        self.up2 = Up(channels[3], channels[2] // factor, bilinear)
        self.up3 = Up(channels[2], channels[1] // factor, bilinear)
        self.up4 = Up(channels[1], cfg.MODEL.FEATURE_DIM, bilinear) #128
        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x