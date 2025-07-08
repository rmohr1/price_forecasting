from typing import Sequence

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size: int=288, hidden_size: int=64,
        num_layers: int=1, dropout: float=0.0,)->nn.Module:
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout, batch_first=True,)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size: int=9, hidden_size: int=64,
        num_layers: int=1, dropout: float=0.0,)->nn.Module:
        super().__init__()

        self.lstm = nn.LSTM(output_size=output_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout, batch_first=True,)

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        out, _ = self.lstm(x, (hidden, cell))
        predictions = self.output_layer(out)
        return predictions

class EncoderDecoder(nn.Module):
    def __init__(
        self,
        input_size: int=288,
        quantiles: Sequence=[0.05, 0.5, 0.95],
        hidden_size: int=64,
        dropout_encoder: float=0.0,
        dropout_decoder: float=0.0,
        **kwargs,
    )->nn.Module:

        super().__init__()

        self.quantiles = quantiles
        out_size = len(quantiles)
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size,
                               dropout = dropout_encoder)

        self.decoder = Decoder(output_size=out_size, hidden_size=hidden_size,
                               dropout = dropout_decoder)
    def forward(self, x):
        hidden, cell = self.encoder(x)
        decoder_in = None
        out, _ = self.lstm(x)
        out = self.output_layer(out)
        return out.squeeze(-1)
