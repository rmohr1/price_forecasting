from price_forecasting.train.train_lstm_model import load_and_train
from price_forecasting.config import TEST_DIR


def test_load_and_train(tmp_path):
    config = {
        "hidden_size": 32,
        "quantiles": [0.05, 0.5, 0.95],
        "learning_rate": 0.01,
        "batch_size": 100,
        "epochs": 1,
        "dropout": 0.1,
        "data_source": 'v1',
    }

    load_and_train(tmp_path, config=config)

    assert tmp_path.exists()
    assert (tmp_path / 'config.yaml').exists
    assert (tmp_path / 'model_wts.pt').exists
    assert (tmp_path / 'y_pred.npy').exists

