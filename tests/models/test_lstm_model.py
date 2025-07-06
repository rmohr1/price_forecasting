import torch

from price_forecasting.models.quantile_lstm import QuantileLSTM


def test_lstm_output_shape():
    model = QuantileLSTM(input_size=288, hidden_size=64, quantiles=[0.1, 0.5, 0.9])
    x = torch.randn(8, 288)   # batch, seq_len, input_size
    y = model(x)
    assert y.shape == (8, 3)