import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

import gc
import time
from datetime import timedelta


class VGGBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_ratio=4):
        super().__init__()
        mid_channels = out_channels // bottleneck_ratio
        self.reduce = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.expand = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.reduce(x)))
        out = self.relu(self.bn2(self.conv(out)))
        out = self.bn3(self.expand(out))
        out += identity
        out = self.relu(out)
        return out


class VGG_BN(nn.Module):
    def __init__(self, vgg_name, num_classes=10, dataset='cifar10'):
        super(VGG_BN, self).__init__()
        self.dataset = dataset.lower()

        self.features = self._make_layers_with_residuals(vgg_name)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers_with_residuals(self, vgg_name):
        cfg = {
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
        }

        model = cfg[vgg_name]
        layers = []
        in_channels = 3
        i = 0

        while i < len(model):
            # for x in cfg[vgg_name]:
            x = model[i]
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                i += 1
            else:
                # Check if next layer is also conv (not 'M')
                if (i + 1 < len(model)) and (model[i + 1] != 'M'):
                    layers += [VGGBottleneckBlock(in_channels, x)]
                    in_channels = x
                    i += 2  # skip next conv as it's residual
                else:
                    layers += [
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True)
                    ]
                    in_channels = x
                    i += 1
        return nn.Sequential(*layers)
