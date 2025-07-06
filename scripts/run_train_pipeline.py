import argparse

from price_forecasting.config import MODELS_DIR
from price_forecasting.train.train_pipeline import load_and_train

parser = argparse.ArgumentParser()
parser.add_argument(
    "model_dir",  # No flag here, just the name
    type=str,
    help="name of the model directory"
)
args = parser.parse_args()
dir_name = args.model_dir
path = MODELS_DIR / dir_name

load_and_train(path)