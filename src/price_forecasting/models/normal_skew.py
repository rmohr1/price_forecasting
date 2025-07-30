from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Transform, TransformedDistribution, constraints


class SinhArcsinhTransform(Transform):
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, skew=0.0, tailweight=1.0, cache_size=1):
        super().__init__(cache_size=cache_size)
        self.skew = skew
        self.tailweight = tailweight

    def __eq__(self, other):
        return isinstance(other, SinhArcsinhTransform)

    def _call(self, x):
        return torch.sinh(torch.asinh(x) + self.skew) * self.tailweight

    def _inverse(self, y):
        return torch.sinh(torch.arcsinh(y / self.tailweight) - self.skew)

    def log_abs_det_jacobian(self, x, y):
        # dy/dx = tailweight * cosh(asinh(x) + skew) / sqrt(1 + x^2)
        sinh_arg = torch.asinh(x) + self.skew
        numerator = torch.cosh(sinh_arg)
        denominator = torch.sqrt(1 + x**2)
        return torch.log(torch.abs(self.tailweight * numerator / denominator))

def get_dist(params):
    """takes a set of parameters in dist and returns a torch distribution object
    """
    skew = params["skew"].squeeze(-1)
    loc = params["loc"].squeeze(-1)
    scale = params["scale"].squeeze(-1)
    tailweight = params["tailweight"].squeeze(-1)

    dist = Normal(loc, scale)
    transform = SinhArcsinhTransform(skew=skew, tailweight=tailweight)
    dist = TransformedDistribution(dist, transform)
    return dist



class NormalSkew(nn.Module):
    def __init__(
        self,
        input_size: int=288*2,
        output_size: int=288,
        hidden_size: int=64,
        num_layers: int=2,
        dropout: float=0.2,
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

        self.skew_head = nn.Linear(hidden_size, 1)
        self.tailweight_head = nn.Linear(hidden_size, 1)
        self.loc_head = nn.Linear(hidden_size, 1)
        self.scale_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)

        skew = self.skew_head(out).clamp(min=-3.0, max=3.0)
        loc = self.loc_head(out)
        tailweight = 1e-3 + F.softplus(self.tailweight_head(out)).clamp(max=5.0)
        scale = 1e-3 + F.softplus(self.scale_head(out))

        
        dist_params = {
            "skew":skew[:,-self.output_size:,:],
            "loc":loc[:,-self.output_size:,:],
            "tailweight":tailweight[:,-self.output_size:,:],
            "scale":scale[:,-self.output_size:,:],
        }

        return dist_params

    def loss(
            self,
            params: Sequence,
            target: Sequence,
        ):
        """Calculate log likelihood loss function using a student t distribution mix.

        Args:
            dist: probability dist predictions from model
            target: target value to be trained on

        Returns:
            loss: 

        """
        #skew = dist["skew"].squeeze(-1)
        #loc = dist["loc"].squeeze(-1)
        #scale = dist["scale"].squeeze(-1)
        #tailweight = dist["tailweight"].squeeze(-1)

        #dist = Normal(loc, scale)
        #transform = SinhArcsinhTransform(skew=skew, tailweight=tailweight)
        #dist = TransformedDistribution(dist, transform)
        dist = get_dist(params)
        y_true = target[:, -self.output_size:]
        alpha = 3.0
        beta = 0.1

        #residual = torch.abs(dist.mean() - y_true)
        #weight = 1.0 + alpha * torch.tanh(beta * residual)
        weight = 1.0 + alpha * torch.tanh(beta * torch.abs(y_true))
        weighted_loss = -dist.log_prob(y_true) * weight

        #return -dist.log_prob(target[:, -self.output_size:])
        return weighted_loss