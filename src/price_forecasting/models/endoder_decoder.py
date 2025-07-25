from typing import Sequence

import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        input_size: int=7,
        quantiles: Sequence=[0.05, 0.5, 0.95],
        hidden_size: int=64,
        dropout: float=0.2,
        target_len=288,
        **kwargs,
    )->nn.Module:

        super().__init__()

        self.quantiles = quantiles
        output_size = len(quantiles)
        self.output_size = output_size
        self.target_len = target_len

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=1,
                               batch_first=True)
        
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers=1,
                               batch_first=True)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # get context from encoder
        _, (hidden, cell) = self.encoder(x)

        # first decoder initialization [batch_size, 1, output_size]
        decoder_input = torch.zeros(x.size(0), 1, self.output_size)

        outputs = torch.Tensor(x.size(0), self.target_len, self.output_size)
        for i in range(self.target_len):
            decoder_output, _ = self.decoder(decoder_input, (hidden, cell))

            # implement dropout before output layer
            decoder_output = self.dropout_layer(decoder_output)

            # Shape (batch_size, output_size)
            output = self.output_layer(decoder_output.squeeze(1))
            outputs[:, i, :] = output #save output point

            decoder_input = output.unsqueeze(1) #feed back output to input

        out_dict = {}
        for i, q in enumerate(self.quantiles):
            out_dict[str(q)] = outputs[:, :, i]
        return out_dict

    def loss(
        self,
        preds: Sequence,
        target: Sequence,
    ):
        """Calculate quantile loss function.

        Args:
            preds: quantile predictions from model
            target: target value to be trained on

        Returns:
            loss: 
        """
        losses = []
        for q_str, pred in preds.items():
            q = float(q_str)
            errors = target - pred
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss 