from pathlib import Path

# Base project directory. Dynamic environment resolving can be added later
PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data" #data storage
RAW_DATA_DIR = DATA_DIR / "raw" #raw data storage
PROCESSED_DATA_DIR = DATA_DIR / "processed" #processed data storage
MODELS_DIR = PROJECT_DIR / "models" #trained models and parameters
