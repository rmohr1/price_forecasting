import torch

from price_forecasting.models.endoder_decoder import EncoderDecoder
from price_forecasting.models.quantile_lstm import QuantileLSTM


def test_lstm_output_shape():
    model = QuantileLSTM(input_size=10, hidden_size=64, quantiles=[0.1, 0.5, 0.9])
    x = torch.randn(8, 10)   # batch, seq_len, input_size
    y = model(x)
    assert y.shape == (8, 3)

def test_encoder_decoder_output_shape():
    model = EncoderDecoder(input_size=8, hidden_size=64, quantiles=[0.1, 0.5, 0.9],
                           target_len=288)
    x = torch.randn(8, 288, 8)   # batch, seq_len, input_size
    y = model(x)
    assert y.shape == (8, 288, 3)