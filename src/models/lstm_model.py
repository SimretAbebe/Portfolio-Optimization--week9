import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def scale_data(train_series, test_series):
    """Scale train and test series to [0, 1] using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
    test_scaled = scaler.transform(test_series.values.reshape(-1, 1))
    return train_scaled, test_scaled, scaler


def create_sequences(data, window_size=60):
    """Create sliding‑window sequences for LSTM.
    Returns X of shape (samples, window_size, 1) and y of shape (samples,).
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i, 0])
        y.append(data[i, 0])
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    return X, y


def build_lstm_model(window_size, units=50, dropout_rate=0.2):
    """Define a two‑layer LSTM network.
    - First LSTM returns sequences (so we can stack another LSTM).
    - Dropout helps prevent over‑fitting.
    - Final dense layers map to a single price output.
    """
    model = Sequential(
        [
            LSTM(units, return_sequences=True, input_shape=(window_size, 1)),
            Dropout(dropout_rate),
            LSTM(units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(25, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_lstm(model, X_train, y_train, epochs=25, batch_size=32, validation_split=0.1):
    """Fit the LSTM with early stopping.
    Returns the Keras History object.
    """
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=1,
    )
    return history


def forecast_lstm(model, train_scaled, test_scaled, window_size, scaler):
    """Generate forecasts for the test period using a rolling window.
    Returns predictions in the original price scale.
    """
    # Concatenate the tail of training data with the whole test set
    full_data = np.concatenate([train_scaled[-window_size:], test_scaled], axis=0)
    predictions_scaled = []
    for i in range(window_size, len(full_data)):
        window = full_data[i - window_size : i, 0].reshape(1, window_size, 1)
        pred = model.predict(window, verbose=0)
        predictions_scaled.append(pred[0, 0])
    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions_scaled)
    return predictions.flatten()


def plot_training_history(history):
    """Plot training and validation loss curves."""
    plt.figure()
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title("LSTM Training History")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


from src.evaluation import calculate_metrics, compare_models
