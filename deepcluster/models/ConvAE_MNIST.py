from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAE_MNIST(nn.Module):
    def __init__(self, embed_size: int = 84, final_activation: nn.Module = None):
        super(ConvAE_MNIST, self).__init__()
        self.encoder_conv = nn.Sequential(nn.Sequential(nn.Conv2d(1, 32, 4, stride = 1), nn.MaxPool2d(2), nn.ReLU()),
                                          nn.Sequential(nn.Conv2d(32, 32, 5, stride = 1), nn.MaxPool2d(2), nn.ReLU()),
                                          nn.Sequential(nn.Conv2d(32, 64, 4, stride = 1), nn.ReLU()))
        #output (B, 64, 1, 1)
        self.encoder_linear = nn.Sequential(nn.Sequential(nn.Linear(64, 120), nn.ReLU()),
                                            nn.Sequential(nn.Linear(120, embed_size)))

        if final_activation is not None:
            self.encoder_linear[-1].append(final_activation)
        
        self.decoder_linear = nn.Sequential(nn.Sequential(nn.Linear(embed_size, 120), nn.ReLU()),
                                            nn.Sequential(nn.Linear(120, 64), nn.ReLU()))
        
        self.decoder_conv = nn.Sequential(nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride = 1), nn.ReLU()),
                                          nn.Sequential(nn.ConvTranspose2d(32, 32, 6, stride = 2), nn.ReLU()),
                                          nn.Sequential(nn.ConvTranspose2d(32, 1, 6, stride = 2)))
        
    def forward(self, x):
        embedding = self.encoder_linear(self.encoder_conv(x).view(-1, 64))
        reconstruction = self.decoder_conv(self.decoder_linear(x).view(-1, 64, 1, 1))

        return reconstruction, embedding

