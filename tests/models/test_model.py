import torch

from price_forecasting.models.endoder_decoder import EncoderDecoder
from price_forecasting.models.quantile_lstm import QuantileLSTM
from price_forecasting.models.studentt_lstm import StudentTLSTM
from price_forecasting.models.normal_skew import NormalSkew


def test_lstm_output_shape():
    model = QuantileLSTM(input_size=10, hidden_size=64, quantiles=[0.1, 0.5, 0.9])
    x = torch.randn(5, 10, 10)   # batch, seq_len, input_size
    y = model(x)
    assert(len(y.keys()) == 3)
    assert(y['0.5'].shape == (5, 10))

def test_encoder_decoder_output_shape():
    model = EncoderDecoder(input_size=8, hidden_size=64, quantiles=[0.1, 0.5, 0.9],
                           target_len=288)
    x = torch.randn(8, 288, 8)   # batch, seq_len, input_size
    y = model(x)
    assert(len(y.keys()) == 3)
    assert(y['0.5'].shape == (8, 288))

def test_studentT_output_shape():
    model = StudentTLSTM(input_size=8, hidden_size=64, N_mix=3)
    x = torch.randn(8, 288, 8)   # batch, seq_len, input_size
    y = model(x)
    assert(len(y.keys()) == 4) # 4 entries to define distribution
    assert(y['df'].shape == (8, 288, 3)) #batch, seq_len, N_mix

def test_normal_skew_output_shape():
    model = NormalSkew(input_size=8, output_size=288, hidden_size=64)
    x = torch.randn(8, 288*2, 8)   # batch, seq_len, input_size
    y = model(x)
    assert(len(y.keys()) == 4) # 4 entries to define distribution
    assert(y['skew'].shape == (8, 288, 1)) #batch, seq_len