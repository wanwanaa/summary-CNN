import torch
import torch.nn as nn
from models import *


class Encoder_cnn(nn.Module):
    def __init__(self, embeds, config):
        super().__init__()
        self.embeds = embeds
        self.embedding_dim = config.embedding_dim
        self.hidden_size = config.hidden_size
        self.n_layer = config.n_layer
        self.t_len = config.t_len

        # convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(4)
        )

        # Linear
        size = int(8*self.hidden_size/16*(self.t_len//16))
        self.linear_out = nn.Linear(size, self.hidden_size)

    def forward(self, x):
        # e(batch, 1(in channel), t_len, hidden_size)
        e = self.embeds(x).unsqueeze(1)
        # (batch, out channel, t_len/2, hidden_size/2)
        out = self.conv1(e)
        # (batch, out channel, t_len/4, hidden_size/4)
        out = self.conv2(out)

        out = out.view(x.size(0), -1)
        out = self.linear_out(out).view(1, -1, self.hidden_size)
        out = out.repeat(self.n_layer, 1, 1)
        return out