# -*- coding: utf-8 -*-
from collections import OrderedDict

import torch.nn as nn

__all__ = ["feedforward"]

# Fully connected neural network with one hidden layer
class FeedForward(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, num_classes=10):
        super(FeedForward, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.sigmoid = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size, num_classes)   

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        out = self.l1(x)
        out = self.sigmoid(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out 
    
    def _eval_layers(self):
        self.l1.eval()
        self.l2.eval()

def feedforward(conf):
    """Constructs a 1 layer fully connected feedforward model."""
    return FeedForward()

def count_parameters():
    model = FeedForward()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)