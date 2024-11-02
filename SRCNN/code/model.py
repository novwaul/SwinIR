import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, channels, kernel_sizes):
        super().__init__()
        L = list()
        for i in range(len(kernel_sizes)):
            L.append(nn.Conv2d(channels[i], channels[i+1], kernel_sizes[i], padding=1))
            if i < len(kernel_sizes)-1:
                L.append(nn.ReLU())
        self.model = nn.Sequential(*L)
    
    def forward(self, img):
        return self.model(img)
