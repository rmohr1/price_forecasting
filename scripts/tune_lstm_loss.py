import os
import shutil

import numpy as np
import optuna
import pandas as pd
import yaml

from price_forecasting.config import MODELS_DIR, PROCESSED_DATA_DIR
from price_forecasting.train.train_pipeline import load_and_train
from price_forecasting.utils.scoring_tools import get_mean_crps

# optuna settings
N_JOBS = 4
N_TRIALS = 50

# directory structuring
MODEL_DIR = MODELS_DIR / 'LSTM_loss_tuning'
os.makedirs(MODEL_DIR, exist_ok=True)

# set train/test data source
DATA_SOURCE = PROCESSED_DATA_DIR / 'v1'

# load y_test data and format as numpy array
y_test = pd.read_parquet(DATA_SOURCE / 'y_test.pqt').to_numpy()
y_test = y_test.reshape([-1])


# define hyperparameter tuning structure
quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999]

def train_and_score_model(hidden_size, learning_rate, dropout, batch_size, trial_n):
    config = {
        "hidden_size": hidden_size,
        "quantiles": quantiles,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": 50,
        "dropout": dropout,
        "epoch_grade": "loss",
        "data_source": 'v1',
    }
    TRIAL_DIR = MODEL_DIR / ("trial"+str(trial_n))
    score = load_and_train(TRIAL_DIR, config=config)
    return score

def objective(trial):
    hidden_size = trial.suggest_int("hidden_size", 50, 75)
    #batch_size = trial.suggest_int("batch_size", 8, 64)
    batch_size = 8
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.3, 0.5)
    
    score = train_and_score_model(hidden_size, learning_rate, 
                                 dropout, batch_size, trial.number)
    return score

# run optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)

# save trial number and score for best trial
best = study.best_trial.number
score = study.best_trial.values[0]

# save to optuna.yaml
out_dict = {"best": "trial"+str(best), "score": score}
with open(MODEL_DIR / 'optuna.yaml', mode='w') as f:
    yaml.dump(out_dict, f)

#copy best trial parameters to top level model directory
TRIAL_DIR = MODEL_DIR / ("trial"+str(best))
shutil.copytree(TRIAL_DIR, MODEL_DIR, dirs_exist_ok=True)