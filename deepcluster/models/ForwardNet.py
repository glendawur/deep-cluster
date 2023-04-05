import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f

class ForwardNet(nn.Module):
    """
    Dense Forward Net
    """

    def __init__(self, input_dim: int, output_dim: int, intermediate: list, activation_f: list, batch_norm: bool = True):
        """
        Number of layers: len(intermediate)+1

        """
        super(ForwardNet, self).__init__()

        self.layers = nn.Sequential()

        layers = [nn.Linear(i, j) for i, j in zip([input_dim] + intermediate, intermediate + [output_dim])]

        for layer, activation in zip(layers, activation_f):
            self.layers.append(nn.Sequential(layer, activation))

        if len(self.layers) < len(layers):
            self.layers.extend([nn.Sequential(layer) for layer in layers[len(self.layers):]])

        if batch_norm == True:
            for i in range(len(intermediate)):
                self.layers[i].insert(1, nn.BatchNorm1d(self.layers[i][0].out_features))

    def forward(self, x):
        return self.layers(x)