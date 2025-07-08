# price_forecasting

Forecasting real-time electricity prices at CAISO nodes using day-ahead market data and pytorch LSTM neural network models.

## Objectives

This is not a production-grade system, but rather a scoped technical demonstration to showcase:
- Clean project structure
- Modular pipelines for data and modeling
- Model predictions of real-time nodal price with uncertainty quantification
- Probabilistic evaluation metrics (CRPS, coverage, spread)
- Comparative model analysis

## Highlights

### Data Pipeline 
Modular scripts for pulling nodal data using CAISO OASIS API. Cleaning scripts and option for version control.

### Modular Neural Network Models
A baseline LSTM model included with functionality to define and plug in new models.

### Evaluation Metrics
Tools to calculate CRPS and notebooks to load and compare coverage, spread, and bias between models. 

### Hyperparameter Tuning
Implementation of model tuning via Optuna

## Project Structure

```text
price_forecasting/
├── reports/ # Directory for clean modeling reports
│ ├── html # HTML formatted reports
├── notebooks/ # Visualizations, evaluation notebooks
├── scripts/ # Run scripts (training, tuning, results generation)
├── models/ # Saved weights, predictions, configs per model
├── data/ # Raw and processed data (ignored in git)
├── src/price_forecasting/
│ ├── data/ # Pulling and cleaning logic
│ ├── models/ # pytorch model architectures
│ ├── train/ # Training methods and pipelines
│ ├── utils/ # Shared utilities and scoring functions
│ ├── config.py # Relative paths and configurations
├── tests/ # Unit tests for key functionality
├── environment.yml # Conda environment
├── pyproject.toml # Project metadata
└── README.md
```

## Getting Started

To recreate the environment:

```bash
conda env create -f environment.yml
conda activate price_forecasting
```

To view summary reports load the HTML files in reports/html