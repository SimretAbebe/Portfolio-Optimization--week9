import pandas as pd
from statsmodels.tsa.stattools import adfuller


def run_adf_test(series, name):
    result = adfuller(series.dropna())

    test_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    is_stationary = p_value < 0.05

    print(f"--- {name} ---")
    print(f"  Test Statistic : {test_statistic:.4f}")
    print(f"  P-Value        : {p_value:.6f}")
    print(f"  Critical Values:")
    for key, value in critical_values.items():
        print(f"    {key}: {value:.4f}")

    if is_stationary:
        print(f"STATIONARY (p={p_value:.6f} < 0.05)")
    else:
        print(f"NOT STATIONARY (p={p_value:.6f} >= 0.05)")
    print()

    return {
        "Series": name,
        "Test Statistic": round(test_statistic, 4),
        "P-Value": round(p_value, 6),
        "Stationary": is_stationary
    }


def run_adf_for_all(close_prices, daily_returns, tickers):
    results = []

    print("=" * 60)
    print("ADF TEST ON CLOSING PRICES")
    print("=" * 60)
    for ticker in tickers:
        res = run_adf_test(close_prices[ticker], f"{ticker} Close Price")
        results.append(res)

    print("=" * 60)
    print("ADF TEST ON DAILY RETURNS")
    print("=" * 60)
    for ticker in tickers:
        res = run_adf_test(daily_returns[ticker], f"{ticker} Daily Returns")
        results.append(res)

    return pd.DataFrame(results)
