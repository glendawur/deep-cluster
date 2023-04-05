from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNetEmbedding(nn.Module):
    def __init__(self, embed_size: int = 84, final_activation: nn.Module = None):
        super(LeNetEmbedding, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 5), nn.MaxPool2d(2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.MaxPool2d(2), nn.ReLU())
        self.fc1   = nn.Sequential(nn.Linear(16*4*4, 120), nn.ReLU())
        self.fc2   = nn.Sequential(nn.Linear(120, embed_size))
        if final_activation is not None:
            self.fc2.append(final_activation)

    def forward(self, x):
        conv_out = self.conv2(self.conv1(x))
        linear_in = conv_out.view(conv_out.size(0), -1)
        out = self.fc2(self.fc1(linear_in))
        return out