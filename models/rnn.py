import torch
import torch.nn as nn
from models import *


class Embeds(nn.Module):
    def __init__(self, config, embedding=None):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        if embedding:
            self.embeds = nn.Embedding.from_pretrained(embedding)
        else:
            self.embeds = nn.Embedding(self.vocab_size, self.embedding_dim)

    def forward(self, x):
        """
        :param x: (batch, t_len)
        :return: (batch, t_len, embedding_dim
        """
        return self.embeds(x)


class Encoder(nn.Module):
    def __init__(self, embeds, config):
        super().__init__()
        self.embeds = embeds
        self.n_layer = config.n_layer
        self.cell = config.cell
        self.bidirectional = config.bidirectional
        self.hidden_size = config.hidden_size

        if config.cell == 'lstm':
            self.rnn = nn.LSTM(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_size,
                num_layers=config.n_layer,
                batch_first=True,
                dropout=config.dropout,
                bidirectional=config.bidirectional
            )

        else:
            self.rnn = nn.GRU(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_size,
                num_layers=config.n_layer,
                batch_first=True,
                dropout=config.dropout,
                bidirectional=config.bidirectional
            )

    def forward(self, x):
        """
        :param x:(batch, t_len)
        :return: gru_h(n_layer, batch, hidden_size) lstm_h(h, c)
                  out(batch, t_len, hidden_size)
        """
        e = self.embeds(x)
        # out (batch, time_step, hidden_size*bidirection)
        # h (batch, n_layers*bidirection, hidden_size)
        encoder_out, h = self.rnn(e)

        if self.bidirectional:
            encoder_out = encoder_out[:, :, :self.hidden_size] + encoder_out[:, :, self.hidden_size:]
        if self.cell == 'lstm':
            h = (h[0][::2].contiguous(), h[1][::2].contiguous())
        else:
            h = h[:self.n_layer]
        return h, encoder_out


class Decoder(nn.Module):
    def __init__(self, embeds, config):
        super().__init__()
        self.embeds = embeds
        self.attn_flag = config.attn_flag
        self.cell = config.cell
        self.intra_decoder = config.intra_decoder
        self.cnn = config.cnn
        self.rl = config.rl

        if config.cell == 'lstm':
            self.rnn = nn.LSTM(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_size,
                num_layers=config.n_layer,
                batch_first=True,
                dropout=config.dropout,
            )
        else:
            self.rnn = nn.GRU(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_size,
                num_layers=config.n_layer,
                batch_first=True,
                dropout=config.dropout,
            )

        if config.attn_flag == 'bahdanau':
            self.attention = Bahdanau_Attention(config)
        elif config.attn_flag == 'luong':
            self.attention = Luong_Attention(config)
        else:
            self.attention = None

        # cnn prob
        if config.cnn == 2:
            self.linear_enc = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.SELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
            self.linear_h = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.SELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
            # cat (LSTM, CNN) -> vector ues to compute prob
            self.linear_enc_cnn = nn.Sequential(
                nn.Linear(config.hidden_size*2, config.hidden_size),
                nn.SELU(),
                nn.Linear(config.hidden_size, config.hidden_size)
            )
            self.sigmoid = nn.Sigmoid()

        # intra-decoder
        if self.intra_decoder:
            self.intra_attention = Luong_Attention(config)
            self.linear_intra = nn.Linear(config.hidden_size*2, config.hidden_size)

    def forward(self, x, h, encoder_output, cnn_out, outs):
        """
        :param x: (batch, 1) decoder input
        :param h: (batch, n_layer, hidden_size)
        :param encoder_output: (batch, t_len, hidden_size) encoder hidden state
        :param cnn_out: (batch, t_len, hidden_size)
        :return: attn_weight (batch, 1, time_step)
                  out (batch, 1, hidden_size) decoder output
                  h (batch, n_layer, hidden_size) decoder hidden state
        """
        attn_weights = None
        e = self.embeds(x).unsqueeze(1) # (batch, 1, embedding_dim)
        if self.attn_flag == 'bahdanau':
            if self.cell == 'lstm':
                attn_weights, e = self.attention(e, h[0], encoder_output)
            else:
                attn_weights, e = self.attention(e, h, encoder_output)
        out, h = self.rnn(e, h)
        if self.rl:
            baseline = out
        else:
            baseline = None

        # cnn prob
        if self.cnn == 2:
            # (batch, t_len, hidden_size)
            encoder = self.linear_enc(encoder_output)
            if self.cell == 'lstm':
                # (batch, hidden_size, 1)
                h_cnn = self.linear_enc(h[0][-1]).unsqueeze(2)
            else:
                h_cnn = self.linear_enc(h[-1]).unsqueeze(2)

            # (batch, t_len, 1)
            # prob = torch.bmm(encoder, h_cnn)
            # prob = self.sigmoid(prob)

            # cat(lstm,cnn)
            vector = torch.cat(encoder, cnn_out)
            vector = self.linear_enc_cnn(vector)
            prob = torch.bmm(vector, h_cnn)
            prob = self.sigmoid(prob)

            encoder_output = prob*encoder_output + (1-prob)*cnn_out

        if self.attn_flag == 'luong':
            attn_weights, out = self.attention(out, encoder_output)
        if self.attn_flag == 'multi':
            attn_weights, out = self.attention(h[0].transpose(0, 1), encoder_output)
        if self.intra_decoder:
            attn_weights, c = self.intra_attention(out, outs)
            out = self.linear_intra(torch.cat((out, c), dim=-1))

        return attn_weights, baseline, out, h