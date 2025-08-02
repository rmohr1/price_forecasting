from typing import Sequence

import torch.nn as nn
import torch


class QuantileLSTM(nn.Module):
    def __init__(
        self,
        input_size: int=288,
        output_size: int=288,
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
        self.output_size = output_size #the number of data points kept after warmup

        self.quantiles = quantiles
        self.output_layer = nn.Linear(hidden_size, len(quantiles))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.output_layer(out)
        out_dict = {}
        for i, q in enumerate(self.quantiles):
            out_dict[str(q)] = out[:, -self.output_size:, i]
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
            output_size = pred.shape[1]
            errors = target[:, -output_size:] - pred
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.cat(losses, dim=1).sum(dim=1)
        return loss

    def epoch_score(self, test_loader, device):
        """Evaluate a torch model against test set.

        Args:
            model: torch model to be trained
            test_loader: test set DataLoader object
            device: torch device type (cpu, gpu)
    
        Returns:
            mean quantile loss over test set

        """
        self.eval()
        total_loss = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = self(x)
                loss = self.loss(preds, y)
                total_loss.append(loss)
        total_loss = torch.cat(total_loss)
        return total_loss.mean()