import numpy as np

from price_forecasting.train.train_pipeline import load_and_train


def test_load_and_train_LSTM(tmp_path):
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
    assert (tmp_path / 'y_pred.npz').exists
    assert (tmp_path / 'y_scaler.pkl').exists

    with np.load(tmp_path / 'y_pred.npz') as y_pred:
        assert(y_pred['0.5'].shape == (112, 288))

def test_load_and_train_EncoderDecoder(tmp_path):
    config = {
        "model": "EncoderDecoder",
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
    assert (tmp_path / 'y_pred.npz').exists
    assert (tmp_path / 'y_scaler.pkl').exists

    with np.load(tmp_path / 'y_pred.npz') as y_pred:
        assert(y_pred['0.5'].shape == (112, 288))

def test_load_and_train_StudentTMixture(tmp_path):
    config = {
        "model": "StudentTLSTM",
        "hidden_size": 32,
        "learning_rate": 0.01,
        "batch_size": 100,
        "epochs": 1,
        "dropout": 0.1,
        "data_source": 'v1',
        "N_mix": 3,
    }

    load_and_train(tmp_path, config=config)

    assert tmp_path.exists()
    assert (tmp_path / 'config.yaml').exists
    assert (tmp_path / 'model_wts.pt').exists
    assert (tmp_path / 'y_pred.npz').exists

    with np.load(tmp_path / 'y_pred.npz') as y_pred:
        assert(y_pred['df'].shape == (112, 288, 3))