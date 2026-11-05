"""Train ARIMA / SARIMA / LSTM, compare MAE-RMSE-MAPE, and run the LSTM window-size ablation."""
import argparse

import numpy as np
import pandas as pd

from models import arima_forecast, lstm_forecast, sarima_forecast
from preprocessing import fit_scaler, load_series, stationarity_tests, train_test_split_series


def metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y), np.asarray(yhat)
    return {
        "MAE": np.mean(np.abs(y - yhat)),
        "RMSE": np.sqrt(np.mean((y - yhat) ** 2)),
        "MAPE%": np.mean(np.abs((y - yhat) / np.where(y == 0, 1e-8, y))) * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--target", default="load")
    ap.add_argument("--horizon", type=int, default=0, help="test points to use (0 = full 20%)")
    ap.add_argument("--ablation", action="store_true")
    args = ap.parse_args()

    s = load_series(args.data, args.time_col, args.target)
    print("Stationarity:", stationarity_tests(s))
    train, test = train_test_split_series(s)
    if args.horizon:
        test = test.iloc[:args.horizon]

    scaler = fit_scaler(train)
    tr_s = scaler.transform(train.values.reshape(-1, 1)).ravel()
    te_s = scaler.transform(test.values.reshape(-1, 1)).ravel()

    results = {
        "ARIMA": metrics(test, arima_forecast(train, len(test))),
        "SARIMA": metrics(test, sarima_forecast(train, len(test))),
    }
    lstm_pred = scaler.inverse_transform(lstm_forecast(tr_s, te_s, window=72).reshape(-1, 1)).ravel()
    results["LSTM"] = metrics(test, lstm_pred)
    print(pd.DataFrame(results).T.round(2))

    if args.ablation:
        rows = {}
        for w in (12, 24, 48, 72):
            p = scaler.inverse_transform(lstm_forecast(tr_s, te_s, window=w).reshape(-1, 1)).ravel()
            m = metrics(test, p)
            rows[w] = {"RMSE": m["RMSE"], "MAE": m["MAE"]}
        print("Window ablation:\n", pd.DataFrame(rows).T.round(2))


if __name__ == "__main__":
    main()
