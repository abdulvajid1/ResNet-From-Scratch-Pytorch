import torch
import torch.nn as nn

"""
Convolution output size calculation:
    Output = floor((Input + 2*padding - kernel_size) / stride) + 1

Architecture.png
"""

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, identity_matrix=None):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.05),
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        
        self.conv_identity = None
        if identity_matrix:
            self.conv_identity = nn.Sequential(
                nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=1)
            )


    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        
        if self.conv_identity:
            x = self.conv_identity(x)
        
        return residual + x
        
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=224, out_channels=64, kernel_size=7, stride=2, padding=3) # (112x112)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2) # (56, 56)
        
        # ResNet layers
        