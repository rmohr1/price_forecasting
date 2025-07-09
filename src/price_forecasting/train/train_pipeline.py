import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from price_forecasting.config import PROCESSED_DATA_DIR
from price_forecasting.models.model_factory import build_model
from price_forecasting.train.trainer import train


# Wrap as DataLoaders
def create_dataloader(x, y, batch_size):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, 
                      shuffle=True, num_workers=0)

def load_and_train(MODEL_DIR, config=None):
    # set up torch
    torch.manual_seed(101)
    np.random.seed(101)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load config file or write config dict if passed
    os.makedirs(MODEL_DIR, exist_ok=True)
    if config is None:
        with open(MODEL_DIR / 'config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    else:
        with open(MODEL_DIR / 'config.yaml', 'w') as f:
            yaml.safe_dump(config, f)

    # define data source path
    DATA_SOURCE = PROCESSED_DATA_DIR / config['data_source']

    # load data
    X_train = pd.read_parquet(DATA_SOURCE / 'X_train.pqt').to_numpy().astype(float)
    y_train = pd.read_parquet(DATA_SOURCE / 'y_train.pqt').to_numpy().astype(float)
    X_test = pd.read_parquet(DATA_SOURCE / 'X_test.pqt').to_numpy().astype(float)
    y_test = pd.read_parquet(DATA_SOURCE / 'y_test.pqt').to_numpy().astype(float)

    # Scale data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    # fit scalers to training data
    X_train = X_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    #transform test data
    X_test = X_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test.reshape(-1, 1))

    pts_per_day = 288
    # reshape train to daily sequence
    n_days_train = len(y_train)//pts_per_day
    X_train = X_train.reshape([n_days_train,pts_per_day,-1])
    y_train = y_train.reshape([n_days_train,pts_per_day])

    # reshape test to daily sequence
    n_days_test = len(y_test)//pts_per_day
    X_test = X_test.reshape([n_days_test,pts_per_day,-1])
    y_test = y_test.reshape([n_days_test,pts_per_day])

    train_loader = create_dataloader(X_train, y_train, config['batch_size'])
    val_loader = create_dataloader(X_test, y_test, config['batch_size'])

    config['input_size'] = X_train.shape[-1]
    model = build_model(config)

    if "epoch_grade" in config:
        eg = config["epoch_grade"]
    else:
        eg = "crps"

    train(model, train_loader, val_loader, y_scaler, config, MODEL_DIR, device, epoch_grade=eg)