# Base PINN model architecture

import torch
import torch.nn as nn

 
# later extend it to support Fourier features, residual connections, etc.


class PINN(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, hidden_layers=4, hidden_units=64, activation=nn.Tanh):
        super().__init__()
        layers = []

        layers.append(nn.Linear(in_dim, hidden_units))
        layers.append(activation())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(activation())

        layers.append(nn.Linear(hidden_units, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)