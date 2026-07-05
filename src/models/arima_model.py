import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


def find_best_arima_order(train_series, seasonal=False, m=1):
    """Automatically discover the optimal ARIMA (or SARIMA) order using pmdarima's auto_arima."""
    model = pm.auto_arima(
        train_series,
        start_p=0,
        start_q=0,
        max_p=5,
        max_q=5,
        d=None,
        seasonal=seasonal,
        m=m,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
    )
    print(f"\n[INFO] Best order found: {model.order}")
    if seasonal:
        print(f"[INFO] Best seasonal order found: {model.seasonal_order}")
    print(f"[INFO] AIC: {model.aic():.2f}")
    return model


def fit_arima(train_series, order):
    """Fit a classic Statsmodels ARIMA model using an explicit order tuple (p, d, q)."""
    model = ARIMA(train_series, order=order)
    fitted = model.fit()
    print(fitted.summary())
    return fitted


def forecast_arima(fitted_model, steps, is_pmdarima=True):
    """Generate a forecast for the next `steps` periods.
    If `is_pmdarima` is True, `fitted_model` is a pmdarima AutoARIMA object; otherwise it's a Statsmodels ARIMA result.
    Returns a tuple (forecast, confidence_interval) where the CI may be None for Statsmodels.
    """
    if is_pmdarima:
        forecast, conf_int = fitted_model.predict(n_periods=steps, return_conf_int=True)
        return forecast, conf_int
    else:
        forecast = fitted_model.forecast(steps=steps)
        return forecast, None


def plot_arima_forecast(train, test, forecast, conf_int=None, title="ARIMA Forecast"):
    """Plot training data, actual test data, and ARIMA forecast (with optional confidence interval)."""
    plt.figure(figsize=(14, 6))
    plt.plot(train.index, train, label="Training", color="#3498db")
    plt.plot(test.index, test, label="Actual Test", color="#2ecc71")
    plt.plot(test.index, forecast, label="ARIMA Forecast", color="#e74c3c", linestyle="--")
    if conf_int is not None:
        plt.fill_between(
            test.index,
            conf_int[:, 0],
            conf_int[:, 1],
            color="#e74c3c",
            alpha=0.15,
            label="95% Confidence Interval",
        )
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()