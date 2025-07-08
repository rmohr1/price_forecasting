from price_forecasting.models.endoder_decoder import EncoderDecoder
from price_forecasting.models.quantile_lstm import QuantileLSTM

MODEL_REGISTRY = {
    "QuantileLSTM": QuantileLSTM,
    "EncoderDecoder": EncoderDecoder,
}