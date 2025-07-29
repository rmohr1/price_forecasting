from typing import Sequence

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MixtureSameFamily, StudentT


class StudentTLSTM(nn.Module):
    def __init__(
        self,
        input_size: int=288*2,
        output_size: int=288,
        hidden_size: int=64,
        num_layers: int=2,
        dropout: float=0.2,
        N_mix: int=2,
        **kwargs
    )->nn.Module:
    
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.output_size = output_size #the number of data points kept after warmup

        self.weight_head = nn.Linear(hidden_size, N_mix)
        self.df_head = nn.Linear(hidden_size, N_mix)
        self.loc_head = nn.Linear(hidden_size, N_mix)
        self.scale_head = nn.Linear(hidden_size, N_mix)

    def forward(self, x):
        out, _ = self.lstm(x)

        weight = F.softmax(self.weight_head(out), dim=-1)
        loc = self.loc_head(out)
        scale = 1e-3 + F.softplus(self.scale_head(out))
        df = 2.1 + F.softplus(self.df_head(out))
        
        dist_params = {
            "df":df[:,-self.output_size:,:],
            "scale":scale[:,-self.output_size:,:],
            "weight":weight[:,-self.output_size:,:],
            "loc":loc[:,-self.output_size:,:],
        }

        return dist_params
        """
        mix = MixtureSameFamily(
            Categorical(probs=weight),
            StudentT(df=df, loc=loc, scale=scale)
        )

        return mix
        """

    def loss(
            self,
            dist: Sequence,
            target: Sequence,
        ):
        """Calculate log likelihood loss function using a student t distribution mix.

        Args:
            dist: probability dist predictions from model
            target: target value to be trained on

        Returns:
            loss: 

        """
        weight = dist["weight"]
        df = dist["df"]
        loc = dist["loc"]
        scale = dist["scale"]

        dist = MixtureSameFamily(
            Categorical(probs=weight),
            StudentT(df=df, loc=loc, scale=scale)
        )

        return -dist.log_prob(target[:, -self.output_size:])