from __future__ import print_function
import torch

import torch.nn as nn

def _layer(in_channels, out_channels, activation=True):
    if activation:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

def _mlplayer(in_channels, out_channels, activation=True):
    if activation:
        return nn.Sequential( 
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True)
        )

    else:
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
        )

class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResLayer, self).__init__()
        self.c1 = _layer(in_channels, out_channels)
        self.c2 = _layer(out_channels, out_channels, activation=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.c1(x)
        h = self.c2(h)
        
        # residual connection
        if x.shape[1] == h.shape[1]:
            h += x

        h = self.activation(h)

        return h


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet, self).__init__()
        self.model = nn.Sequential(
            ResLayer(in_channels, 8),
            ResLayer(8, 8),
            ResLayer(8, 16),
            ResLayer(16, 16),
            ResLayer(16, 16),
        )
        #self.pool = nn.AdaptiveAvgPool2d(1)
        spatial_size = 3
        self.pool = nn.AdaptiveAvgPool2d(spatial_size)
        self.classifier = nn.Linear(16*spatial_size**2, out_channels)
    
    def forward(self, x):
        B = x.shape[0]
        h = self.model(x)
        p = self.pool(h)
        return self.classifier(p.view(B, -1)), p.view(B,-1)

    def name(self):
        return "ResNet"

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, activation=False):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            _mlplayer(in_channels, 500, activation=activation),
            _mlplayer(500, 200, activation=activation),
            _mlplayer(200, 200, activation=activation)
        )
        self.classifier = _mlplayer(200, out_channels, activation=False)
    
    def forward(self, x):
        B = x.shape[0]
        h = self.model(x)
        return self.classifier(h.view(B, -1)), h
    
    def name(self):
        return "MLP"