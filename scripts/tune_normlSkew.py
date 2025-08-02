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
N_TRIALS = 100

# directory structuring
MODEL_DIR = MODELS_DIR / 'normalSkew_tuning'
os.makedirs(MODEL_DIR, exist_ok=True)

# set train/test data source
DATA_SOURCE = PROCESSED_DATA_DIR / 'v1'

# load y_test data and format as numpy array
y_test = pd.read_parquet(DATA_SOURCE / 'y_test.pqt').to_numpy()
y_test = y_test.reshape([-1])


# define hyperparameter tuning structure

def train_and_score_model(hidden_size, learning_rate, weight_decay, alpha, beta, trial_n):
    config = {
        "model": "NormalSkew",
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": 16,
        "epochs": 100,
        "dropout": 0.0,
        "data_source": 'v1',
        "alpha": alpha,
        "beta": beta,
    }
    TRIAL_DIR = MODEL_DIR / ("trial"+str(trial_n))
    score = load_and_train(TRIAL_DIR, config=config)

    return score

def objective(trial):
    hidden_size = trial.suggest_int("hidden_size", 60, 100)
    #learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    learning_rate = 0.001
    weight_decay = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    #dropout = trial.suggest_float("dropout", 0.2, 0.5)
    alpha = trial.suggest_float("alpha", 5.0, 20.0)
    beta = trial.suggest_float("beta", 0.05, 0.3)
    
    score = train_and_score_model(hidden_size, learning_rate, weight_decay, 
                                  alpha, beta, trial.number)
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