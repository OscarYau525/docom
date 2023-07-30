# -*- coding: utf-8 -*-
from collections import OrderedDict

import torch.nn as nn

__all__ = ["cnn3"]

# the exact same 3 layer CNN used in https://github.com/liboyue/beer/blob/master/experiments/mnist/mnist.py
class CNN3(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    def _eval_layers(self):
        # set eval() to layers that does not require gradient
        for _, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Sigmoid) or isinstance(module, nn.ReLU) or isinstance(module, nn.AvgPool2d) or isinstance(module, nn.MaxPool2d):
                module.eval()

def cnn3(conf):
    return CNN3()
