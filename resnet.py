import torch
import torch.nn as nn

"""
Convolution output size calculation:
    Output = floor((Input + 2*padding - kernel_size) / stride) + 1

Architecture.png
"""

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, is_identity=None):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.05),
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        
        self.conv_identity = None
        if is_identity:
            self.conv_identity = nn.Sequential(
                nn.Conv2d(out_channels, out_channels*4, kernel_size=1, stride=1)
            )


    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv_identity(x)
        return residual + x
        
class ResNet(nn.Module):
    def __init__(self, resnet_block):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=224, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
            ) # (224) -> (112) -> (56)
        
    def _create_layer(self, in_channels , num_block, padding, kernel_size=3, is_identity=False):
        resnet_layers = nn.ModuleList()
        for i in range(num_block):
            resnet_layers.append(ResNetBlock(in_channels, in_channels, kernel_size=kernel_size, padding=padding, identity_matrix=is_identity))) 