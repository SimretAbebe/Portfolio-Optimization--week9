# Portfolio Optimization & Time-Series Forecasting

This repository contains time-series forecasting and portfolio optimization tools developed for financial modeling tasks (specifically forecasting Tesla's stock price).

## Repository Structure

```
├── data/                    # Cleaned and processed dataset files
├── notebooks/               # Jupyter notebooks for EDA and Modeling
│   ├── task1_eda.ipynb      # Exploratory Data Analysis
│   └── task2_modeling.ipynb # Time-Series Modeling (ARIMA & LSTM)
├── src/                     # Core python codebase
│   ├── models/
│   │   ├── arima_model.py   # ARIMA parameter selection, training, & forecasting
│   │   └── lstm_model.py    # Deep learning LSTM sequence design & training
│   ├── data_loader.py       # yfinance extraction and preprocessing
│   ├── train_test_split.py  # Chronological dataset splitting utilities
│   ├── evaluation.py        # Centralized model metrics (MAE, RMSE, MAPE)
│   └── risk_metrics.py      # VaR and Sharpe Ratio calculations
├── requirements.txt         # Virtual environment dependencies
└── README.md                # Project documentation
```

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run notebooks**:
   - Use `notebooks/task1_eda.ipynb` to analyze historical price trends.
   - Use `notebooks/task2_modeling.ipynb` to execute the statistical and deep learning forecasts.
