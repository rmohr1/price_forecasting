from typing import Sequence

import scipy.stats as stats
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
        alpha: float=3.0,
        beta: float=0.1,
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

        self.alpha = alpha #weighting scale parameter
        self.beta = beta #weighting range parameter

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
            params: probability dist parameter predictions from model
            target: target values to be trained on

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

        mean = dist.rsample((500,)).mean(dim=0)
        residual = torch.abs(mean - y_true)
        weight = 1.0 + self.alpha * torch.tanh(self.beta * residual)

        #weight = 1.0 + self.alpha * torch.tanh(self.beta * torch.abs(y_true))

        weighted_loss = -dist.log_prob(y_true) * weight

        #return -dist.log_prob(target[:, -self.output_size:])
        return weighted_loss
    
    def epoch_score(self, test_loader, device):
        self.eval()
        y_test = []
        with torch.no_grad():
            y_pred = None
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = self(x)
                if y_pred is None:
                    y_pred = preds
                else:
                    y_pred = {k: torch.cat([y_pred[k], preds[k]], dim=0) for k in y_pred}
                y_test.append(y[:, -self.output_size:])

        y_test = torch.concat(y_test, dim=0).reshape(-1)
        tailweight = y_pred['tailweight'].reshape(-1)
        skew = y_pred['skew'].reshape(-1)
        loc = y_pred['loc'].reshape(-1)
        scale = y_pred['scale'].reshape(-1)

        y_test_scaled = torch.sinh(torch.asinh(y_test / tailweight) - skew)

        PIT = stats.norm.cdf(y_test_scaled, loc, scale)
        score = stats.cramervonmises(PIT, 'uniform').statistic
        return score
