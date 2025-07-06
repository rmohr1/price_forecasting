import os
from datetime import datetime

import numpy as np
import pandas as pd

from price_forecasting.config import CLEANED_DATA_DIR, PROCESSED_DATA_DIR

TIME_COLUMN = 'timestamp'
OUTPUT_DIR = PROCESSED_DATA_DIR / 'v1'


df = pd.read_csv(CLEANED_DATA_DIR / 'model_ready_dataset_5min.csv',
                 parse_dates=[TIME_COLUMN])
df = df.set_index(TIME_COLUMN)
df = df.dropna()

# fitting target: calc diff between RTM and DAM price
df['RTM-DAM'] = df['RTM_PRC'] - df['DAM_PRC']

# Define features and target
target = "RTM-DAM"
features = [
    "DAM_PRC",
    "2DA_LOAD",
    "hour",
    "dayofweek",
    "month",
    "year",
    "is_weekend",
    "is_night",
    "RTM_PRC_lag24hr",
]

# Train/test split based on date and time
cutoff = datetime.fromisoformat('2025-03-01 00:00:00-08:00') #train/test date cutoff

train_df = df[df.index < cutoff]
test_df = df[df.index >= cutoff]

#ensure train set starts at 8:00 UTC
train_df = train_df.iloc[np.where(train_df.index.hour==8)[0][0]:] 

#ensure test set ends at 7:55 UTC
mask = (test_df.index.hour==7)&(test_df.index.minute==55)
test_df = test_df.iloc[:np.where(mask)[0][-1] + 1]

X_train = train_df[features]
y_train = train_df[[target]]

X_test = test_df[features]
y_test = test_df[[target]]

os.makedirs(OUTPUT_DIR, exist_ok=True)

X_train.to_parquet(OUTPUT_DIR / 'X_train.pqt')
y_train.to_parquet(OUTPUT_DIR / 'y_train.pqt')

X_test.to_parquet(OUTPUT_DIR / 'X_test.pqt')
y_test.to_parquet(OUTPUT_DIR / 'y_test.pqt')
