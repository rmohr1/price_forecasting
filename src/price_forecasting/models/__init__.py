from price_forecasting.models.endoder_decoder import EncoderDecoder
from price_forecasting.models.quantile_lstm import QuantileLSTM
from price_forecasting.models.studentt_lstm import StudentTLSTM
from price_forecasting.models.normal_skew import NormalSkew

MODEL_REGISTRY = {
    "QuantileLSTM": QuantileLSTM,
    "EncoderDecoder": EncoderDecoder,
    "StudentTLSTM": StudentTLSTM,
    "NormalSkew": NormalSkew,
}