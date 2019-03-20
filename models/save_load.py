from models.attention import *
from models.rnn import *
from models.seq2seq import *
from models.cnn import *


def build_model(config):
    embeds = Embeds(config)
    encoder = Encoder(embeds, config)
    if config.cnn == 1:
        cnn = Encoder_cnn(embeds, config)
    if config.cnn == 2:
        cnn = Encoder_pos(embeds, config)
    decoder = Decoder(embeds, config)
    model = Seq2seq(encoder, cnn, decoder, config)
    return model


def load_model(config, filename):
    model = build_model(config)
    model.load_state_dict(torch.load(filename, map_location='cpu'))
    return model


def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print('model save at ', filename)