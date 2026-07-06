# Portfolio Optimization & Time-Series Forecasting

This repository contains time-series forecasting and portfolio optimization work for a financial modeling pipeline that combines Tesla forecasting with portfolio construction and backtesting.

## Repository Structure

```
├── data/                    # Cleaned and processed dataset files and forecast outputs
├── notebooks/               # Jupyter notebooks for analysis, modeling, optimization, and backtesting
│   ├── task1_eda.ipynb      # Exploratory Data Analysis
│   ├── task2_modeling.ipynb # Time-Series Modeling (ARIMA & LSTM)
│   ├── task3_forecasting.ipynb # TSLA forecast generation and diagnostics
│   ├── task4_portfolio_optimization.ipynb # Efficient Frontier and portfolio recommendation
│   └── task5_backtesting.ipynb # Strategy backtesting vs. 60/40 benchmark
├── src/                     # Core python codebase
│   ├── models/
│   │   ├── arima_model.py   # ARIMA parameter selection, training, & forecasting
│   │   └── lstm_model.py    # LSTM sequence design, training, & forecasting
│   ├── data_loader.py       # yfinance extraction and preprocessing
│   ├── eda_utils.py         # EDA helpers and visualization utilities
│   ├── evaluation.py        # Centralized model metrics (MAE, RMSE, MAPE)
│   ├── risk_metrics.py      # VaR and Sharpe Ratio calculations
│   ├── stationarity.py      # Augmented Dickey-Fuller and stationarity utilities
│   ├── train_test_split.py  # Chronological dataset splitting utilities
│   └── style_fix.py         # Robust matplotlib style loading across notebooks
├── requirements.txt         # Virtual environment dependencies
└── README.md                # Project documentation
```

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run notebooks**:
   - `notebooks/task1_eda.ipynb`: exploratory data analysis and trend inspection.
   - `notebooks/task2_modeling.ipynb`: ARIMA and LSTM forecasting experiments.
   - `notebooks/task3_forecasting.ipynb`: TSLA forecast generation and forecast cleanup.
   - `notebooks/task4_portfolio_optimization.ipynb`: efficient frontier construction and portfolio recommendation.
   - `notebooks/task5_backtesting.ipynb`: backtesting the recommended strategy vs. a 60% SPY / 40% BND benchmark.

## Notes

- `src/data_loader.py` provides the data extraction and preprocessing pipeline used by all notebooks.
- `src/style_fix.py` ensures Jupyter notebooks can safely apply plotting styles without crashing on local style files.
- The new `task5_backtesting.ipynb` notebook validates the strategy over the 2025-2026 backtest window, comparing model-driven allocation performance against a passive benchmark.

## Recommended Workflow

1. Run `task1_eda.ipynb` to inspect the historical data and confirm data quality.
2. Run `task2_modeling.ipynb` to fit and evaluate forecasting models.
3. Run `task3_forecasting.ipynb` to generate and save the TSLA forecast used as the forecast view.
4. Run `task4_portfolio_optimization.ipynb` to create the efficient frontier and recommend the Max Sharpe strategy.
5. Run `task5_backtesting.ipynb` to compare the recommended strategy against the benchmark and capture performance metrics.
