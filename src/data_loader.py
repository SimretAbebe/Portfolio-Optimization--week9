import yfinance as yf
import pandas as pd


def fetch_stock_data(tickers, start_date, end_date):
    raw_data = yf.download(tickers, start=start_date, end=end_date)
    print(f"[INFO] Downloaded data shape: {raw_data.shape}")
    print(f"[INFO] Date range: {raw_data.index.min()} to {raw_data.index.max()}")
    return raw_data


def extract_close_prices(raw_data, tickers):
    # Safely index tickers to avoid yfinance alphabetical column swap issues
    close_prices = raw_data["Close"][tickers].copy()
    return close_prices


def clean_data(df):
    missing_before = df.isnull().sum().sum()
    # Use modern pandas ffill() and bfill() to avoid deprecation warnings
    df_clean = df.ffill().bfill()
    missing_after = df_clean.isnull().sum().sum()
    print(f"[INFO] Missing values — before: {missing_before}, after: {missing_after}")
    return df_clean


def get_missing_report(df):
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    report = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %": missing_pct.round(2)
    })
    return report


def calculate_daily_returns(df):
    returns = df.pct_change().dropna()
    return returns
