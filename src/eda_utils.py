import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


COLORS = ['#e74c3c', '#2ecc71', '#3498db']


def plot_closing_prices(df, tickers):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for i, ticker in enumerate(tickers):
        axes[i].plot(df.index, df[ticker],
                     color=COLORS[i], linewidth=1.2, label=ticker)
        axes[i].set_title(f"{ticker} — Closing Price",
                          fontsize=14, fontweight='bold')
        axes[i].set_ylabel("Price (USD)", fontsize=11)
        axes[i].legend(loc='upper left', fontsize=11)
        axes[i].grid(True, alpha=0.3)

    axes[2].set_xlabel("Date", fontsize=11)
    fig.suptitle("Historical Closing Prices (2015–2025)",
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()


def plot_normalized_prices(df, tickers):
    normalized = df / df.iloc[0] * 100

    plt.figure(figsize=(14, 6))
    for i, ticker in enumerate(tickers):
        plt.plot(normalized.index, normalized[ticker],
                 label=ticker, linewidth=1.5, color=COLORS[i])

    plt.title("Normalized Closing Prices (Base = 100)",
              fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Normalized Price", fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_daily_returns(daily_returns, tickers):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for i, ticker in enumerate(tickers):
        axes[i].plot(daily_returns.index, daily_returns[ticker],
                     color=COLORS[i], linewidth=0.5, alpha=0.7)
        axes[i].set_title(f"{ticker} — Daily Returns",
                          fontsize=13, fontweight='bold')
        axes[i].set_ylabel("Return", fontsize=11)
        axes[i].axhline(y=0, color='black', linewidth=0.8, linestyle='--')
        axes[i].grid(True, alpha=0.3)

    axes[2].set_xlabel("Date", fontsize=11)
    fig.suptitle("Daily Percentage Returns (2015–2025)",
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()


def plot_return_distributions(daily_returns, tickers):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, ticker in enumerate(tickers):
        mean_val = daily_returns[ticker].mean()
        axes[i].hist(daily_returns[ticker], bins=100, color=COLORS[i],
                     edgecolor='white', alpha=0.75)
        axes[i].set_title(f"{ticker} — Return Distribution",
                          fontsize=13, fontweight='bold')
        axes[i].set_xlabel("Daily Return", fontsize=11)
        axes[i].set_ylabel("Frequency", fontsize=11)
        axes[i].axvline(x=mean_val, color='black', linewidth=1.5,
                        linestyle='--', label=f'Mean: {mean_val:.4f}')
        axes[i].legend(fontsize=9)

    fig.suptitle("Distribution of Daily Returns",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_rolling_statistics(close_prices, daily_returns, tickers, window=21):
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    for i, ticker in enumerate(tickers):
        rolling_mean = close_prices[ticker].rolling(window=window).mean()
        axes[i, 0].plot(close_prices.index, close_prices[ticker],
                        color=COLORS[i], alpha=0.4, linewidth=0.8,
                        label='Actual Price')
        axes[i, 0].plot(rolling_mean.index, rolling_mean,
                        color='black', linewidth=1.5,
                        label=f'{window}-Day Rolling Mean')
        axes[i, 0].set_title(f"{ticker} — Price with {window}-Day Rolling Mean",
                             fontsize=12, fontweight='bold')
        axes[i, 0].set_ylabel("Price (USD)", fontsize=10)
        axes[i, 0].legend(fontsize=9)
        axes[i, 0].grid(True, alpha=0.3)

        rolling_std = daily_returns[ticker].rolling(window=window).std()
        axes[i, 1].plot(rolling_std.index, rolling_std,
                        color=COLORS[i], linewidth=1,
                        label=f'{window}-Day Rolling Volatility')
        axes[i, 1].set_title(f"{ticker} — {window}-Day Rolling Volatility",
                             fontsize=12, fontweight='bold')
        axes[i, 1].set_ylabel("Std Dev of Returns", fontsize=10)
        axes[i, 1].legend(fontsize=9)
        axes[i, 1].grid(True, alpha=0.3)

    fig.suptitle("Rolling Statistics Analysis",
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()


def detect_outliers(daily_returns, tickers, z_threshold=3):
    z_scores = daily_returns.apply(stats.zscore)
    outlier_summary = {}

    for ticker in tickers:
        outlier_mask = abs(z_scores[ticker]) > z_threshold
        outliers = daily_returns[outlier_mask][[ticker]]
        outlier_summary[ticker] = {
            "count": len(outliers),
            "dates": outliers.index.tolist(),
            "worst_day": outliers[ticker].min(),
            "best_day": outliers[ticker].max()
        }

        print(f"\n{'='*50}")
        print(f"OUTLIERS FOR {ticker} (|z-score| > {z_threshold})")
        print(f"{'='*50}")
        print(f"Number of outlier days: {len(outliers)}")
        if len(outliers) > 0:
            print("\nMost negative (worst days):")
            print(outliers.sort_values(by=ticker).head(5).to_string())
            print("\nMost positive (best days):")
            print(outliers.sort_values(by=ticker).tail(5).to_string())

    return z_scores, outlier_summary


def plot_outliers(daily_returns, z_scores, tickers, z_threshold=3):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for i, ticker in enumerate(tickers):
        axes[i].scatter(daily_returns.index, daily_returns[ticker],
                        color='gray', alpha=0.3, s=5, label='Normal')

        outlier_mask = abs(z_scores[ticker]) > z_threshold
        axes[i].scatter(daily_returns.index[outlier_mask],
                        daily_returns[ticker][outlier_mask],
                        color='red', s=25, label='Outlier', zorder=5)

        axes[i].set_title(f"{ticker} — Daily Returns with Outliers",
                          fontsize=13, fontweight='bold')
        axes[i].set_ylabel("Return", fontsize=11)
        axes[i].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        axes[i].legend(loc='upper right', fontsize=9)
        axes[i].grid(True, alpha=0.3)

    axes[2].set_xlabel("Date", fontsize=11)
    fig.suptitle(f"Outlier Detection (|Z-score| > {z_threshold})",
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()
