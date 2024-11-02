import torch
import torch.nn as nn
import math

class DownBlock(nn.Module):
    def __init__(self, channel, keep_channel=False):
        super().__init__()
        out = channel if keep_channel else 2*channel
        self.inner = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel, out, 3, padding=1)
        )

    def forward(self, img):
        return self.inner(img)

class RCABlock(nn.Module):
    def __init__(self, channel, r):
        super().__init__()
        self.inner = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//r, channel, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        x = self.inner(img)
        z = self.attention(x)
        out = torch.mul(x, z) + img
        return out

class UpBlock(nn.Module):
    def __init__(self, channel, d):
        super().__init__()
        B = 40
        self.inner = nn.Sequential(
            *[RCABlock(channel, r=16) for _ in range(B)],
            nn.Conv2d(channel, 4*channel, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channel, channel//d, 1)
        )
    
    def forward(self, img):
        return self.inner(img)

class DualRegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        F = 20
        
        self.head = nn.Conv2d(3, F, 3, padding=1)
        
        self.down1 = DownBlock(F)
        self.down2 = DownBlock(2*F)
        
        self.up1 = UpBlock(4*F, d=2)
        self.up2 = UpBlock(4*F, d=4)
        
        self.tail_4x = nn.Conv2d(2*F, 3, 3, padding=1)
        self.tail_2x = nn.Conv2d(4*F, 3, 3, padding=1)
        self.tail_1x = nn.Conv2d(4*F, 3, 3, padding=1)

        self.dual1 = DownBlock(3, keep_channel=True)
        self.dual2 = DownBlock(3, keep_channel=True)
        self.dual3 = DownBlock(3, keep_channel=True)
    
    def forward(self, img):

        lr_4x = self.head(img)

        lr_2x = self.down1(lr_4x)
        lr_1x = self.down2(lr_2x)
        
        hr_2x = self.up1(lr_1x)
        hr_4x = self.up2(torch.cat((hr_2x, lr_2x), dim=1))

        out = self.tail_4x(torch.cat((hr_4x, lr_4x), dim=1))
        p_2x = self.tail_2x(torch.cat((hr_2x, lr_2x), dim=1))
        p_1x = self.tail_1x(lr_1x)

        d_2x = self.dual1(out)
        d_1x = self.dual2(d_2x)

        return out, p_1x, p_2x, d_1x, d_2x

