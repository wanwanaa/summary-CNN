import torch
import torch.nn as nn


class Luong_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_layer = config.n_layer

        self.linear_in = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.linear_out = nn.Sequential(
            nn.Linear(config.hidden_size*2, config.hidden_size),
            nn.SELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, output, encoder_out):
        """
        :param output: (batch, 1, hidden_size) decoder output
        :param encoder_out: (batch, t_len, hidden_size) encoder hidden state
        :return: attn_weight (batch, 1, time_step)
                  output (batch, 1, hidden_size) attention vector
        """
        out = self.linear_in(output) # (batch, 1, hidden_size)
        out = out.transpose(1, 2) # (batch, hidden_size, 1)
        attn_weights = torch.bmm(encoder_out, out) # (batch, t_len, 1)
        attn_weights = self.softmax(attn_weights.transpose(1, 2)) # (batch, 1, t_len)

        context = torch.bmm(attn_weights, encoder_out) # (batch, 1, hidden_size)
        output = self.linear_out(torch.cat((output, context), dim=2))

        return attn_weights, output


class Bahdanau_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.t_len = config.t_len
        self.linear_add = nn.Sequential(
            nn.Linear(config.hidden_size*2, config.hidden_size),
            nn.ReLU()
        )
        self.attn = nn.Linear(config.hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.linear_out = nn.Linear(config.hidden_size+config.embedding_dim, config.hidden_size)

    def forward(self, x, output, encoder_out):
        """
        :param x:(batch, 1, embedding_dim)
        :param output:(n_layer, batch, hidden_size) decoder hidden state
        :param encoder_out:(batch, time_step, hidden_size) encoder hidden state
        :return: attn_weight (batch, 1, time_step)
                  context (batch, 1, hidden_size) attention vector
        """
        h = output[-1].view(-1, 1, self.hidden_size).repeat(1, self.t_len, 1) # (batch, t_len, hidden_size)
        vector = torch.cat((h, encoder_out), dim=2) # (batch, t_len, hidden_size*2)
        vector = self.linear_add(vector) # (batch, t_len, hidden_size)

        attn_weights = self.attn(vector).squeeze(2) # (batch, t_len)
        attn_weights = self.softmax(attn_weights).unsqueeze(1) # (batch, 1, t_len)

        context = torch.bmm(attn_weights, encoder_out) # (batch, 1, hidden_size)
        context = self.linear_out(torch.cat((context, x), dim=2))

        return attn_weights, context
