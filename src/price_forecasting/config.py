from pathlib import Path

# Base project directory. Dynamic environment resolving can be added later
PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data" #raw and processed data storage location
MODELS_DIR = PROJECT_DIR / "models" #storage for models and training
