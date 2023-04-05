from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNetAE_MNIST(nn.Module):
    def __init__(self, embed_size: int = 84, final_activation: nn.Module = None):
        super(LeNetAE_MNIST, self).__init__()
        self.encoder_conv = nn.Sequential(nn.Sequential(nn.Conv2d(1, 6, 5), nn.MaxPool2d(2), nn.ReLU()),
                                          nn.Sequential(nn.Conv2d(6, 16, 5), nn.MaxPool2d(2), nn.ReLU()))
       
        self.encoder_linear = nn.Sequential(nn.Sequential(nn.Linear(16*4*4, 120), nn.ReLU()),
                                            nn.Sequential(nn.Linear(120, embed_size)))
        
        self.decoder_linear = nn.Sequential(nn.Sequential(nn.Linear(embed_size, 120), nn.ReLU()),
                                            nn.Sequential(nn.Linear(120, 16*4*4), nn.ReLU()))
        
        self.decoder_conv = nn.Sequential(nn.Sequential(nn.Linear(embed_size, 120), nn.ReLU()),
                                     nn.Sequential(nn.Linear(120, 16*4*4), nn.ReLU()))
