import pandas as pd
import numpy as np


TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.02  # 2% annual


def calculate_var(daily_returns, tickers, confidence_levels=None):
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    var_results = []

    for ticker in tickers:
        for cl in confidence_levels:
            var_value = daily_returns[ticker].quantile(1 - cl)
            var_results.append({
                "Asset": ticker,
                "Confidence": f"{cl*100:.0f}%",
                "VaR (Daily)": round(var_value, 4),
                "Interpretation": f"On {cl*100:.0f}% of days, "
                                  f"{ticker} won't lose more than "
                                  f"{abs(var_value)*100:.2f}%"
            })

    return pd.DataFrame(var_results)


def calculate_sharpe_ratio(daily_returns, tickers, risk_free_rate=None):
    if risk_free_rate is None:
        risk_free_rate = DEFAULT_RISK_FREE_RATE

    annual_returns = daily_returns.mean() * TRADING_DAYS_PER_YEAR
    annual_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (annual_returns - risk_free_rate) / annual_volatility

    sharpe_table = pd.DataFrame({
        "Asset": tickers,
        "Annual Return": [f"{annual_returns[ticker]*100:.2f}%" for ticker in tickers],
        "Annual Volatility": [f"{annual_volatility[ticker]*100:.2f}%" for ticker in tickers],
        "Sharpe Ratio": [round(sharpe[ticker], 4) for ticker in tickers]
    }).set_index("Asset")

    return sharpe_table
