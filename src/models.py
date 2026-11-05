"""ARIMA, SARIMA, LSTM and GARCH forecasters with a common fit/forecast interface."""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def arima_forecast(train: pd.Series, steps: int, order=(2, 0, 2)) -> np.ndarray:
    return ARIMA(train, order=order).fit().forecast(steps).values


def sarima_forecast(train: pd.Series, steps: int, order=(1, 0, 1), seasonal=(1, 1, 1, 24)) -> np.ndarray:
    model = SARIMAX(train, order=order, seasonal_order=seasonal,
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False).forecast(steps).values


def make_windows(arr: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i - window:i])
        y.append(arr[i])
    return np.array(X)[..., None], np.array(y)


def build_lstm(window: int):
    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.Input(shape=(window, 1)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def lstm_forecast(train_scaled: np.ndarray, test_scaled: np.ndarray, window: int = 72,
                  epochs: int = 20, batch_size: int = 64) -> np.ndarray:
    """One-step-ahead forecasts over the test period (teacher forcing on true history)."""
    from tensorflow import keras
    X_tr, y_tr = make_windows(train_scaled, window)
    model = build_lstm(window)
    model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    full = np.concatenate([train_scaled[-window:], test_scaled])
    X_te, _ = make_windows(full, window)
    return model.predict(X_te, verbose=0).ravel()


def garch_volatility(train: pd.Series, steps: int) -> np.ndarray:
    """GARCH(1,1) on percentage changes - forecasts load volatility, not level."""
    from arch import arch_model
    returns = 100 * train.pct_change().dropna()
    res = arch_model(returns, vol="GARCH", p=1, q=1).fit(disp="off")
    return np.sqrt(res.forecast(horizon=steps).variance.values[-1])
