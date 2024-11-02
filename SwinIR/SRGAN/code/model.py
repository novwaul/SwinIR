import torch.nn as nn
import math

class ResBlock(nn.Module):
    def __init__(self, channel, kernel, pad):
        super().__init__()
        self.relu = nn.ReLU()
        self.inner_block = nn.Sequential(
            nn.Conv2d(channel, channel, kernel, padding=pad),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel, padding=pad)
        )

    def forward(self, img):
        x = self.relu(img)
        out = self.inner_block(x)
        return x + out

class PixelShuffleBlock(nn.Module):
    def __init__(self, channel, kernel, pad):
        super().__init__()
        scale_factor = 2
        self.inner_block = nn.Sequential(
            nn.Conv2d(channel, channel*(scale_factor**2), kernel, padding=pad), 
            nn.PixelShuffle(scale_factor)
        )
    
    def forward(self, img):
        return self.inner_block(img)

class SRResNet(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        channel = 64
        kernel = 3
        pad = 1

        self.conv = nn.Conv2d(3, channel, kernel, padding=pad)
        self.res_blocks = nn.Sequential(
            *[ResBlock(channel, kernel, pad) for _ in range(9)],
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel, padding=pad),
        )
        self.pixel_shuffle_blocks = nn.Sequential(
            *[PixelShuffleBlock(channel, kernel, pad) for _ in range(int( math.log2(scale_factor) ))],
            nn.Conv2d(channel, 3, kernel, padding=pad)
        )

    def forward(self, img):
        x = self.conv(img)
        out = self.res_blocks(x)
        out = self.pixel_shuffle_blocks(x+out)
        return out

class DisNet(nn.Module):
    def __init__(self, input_size, scale_factor):
        super().__init__()

        self.inner = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(512 * ((scale_factor*input_size//16)**2), 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )
    
    def forward(self, img):
        return self.inner(img)