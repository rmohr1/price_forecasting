from typing import Sequence

import torch.nn as nn
import torch


class QuantileLSTM(nn.Module):
    def __init__(
        self,
        input_size: int=288,
        quantiles: Sequence=[0.05, 0.5, 0.95],
        hidden_size: int=64,
        num_layers: int=2,
        dropout: float=0.2,
        **kwargs
    )->nn.Module:
    
        super(QuantileLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.quantiles = quantiles
        self.output_layer = nn.Linear(hidden_size, len(quantiles))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.output_layer(out)
        out_dict = {}
        for i, q in enumerate(self.quantiles):
            out_dict[str(q)] = out[:, :, i]
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