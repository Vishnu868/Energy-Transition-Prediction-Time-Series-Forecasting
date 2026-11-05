"""Loading, cleaning, scaling and stationarity analysis for electricity load series."""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, kpss


def load_series(path: str, time_col: str, target: str, freq: str = "h") -> pd.Series:
    df = pd.read_csv(path, parse_dates=[time_col])
    df = df.drop_duplicates(subset=time_col).sort_values(time_col).set_index(time_col)
    s = df[target].astype(float).resample(freq).mean()
    return s.interpolate(method="time").dropna()


def stationarity_tests(s: pd.Series) -> dict:
    adf_stat, adf_p, *_ = adfuller(s.values, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(s.values, regression="c", nlags="auto")
    return {
        "ADF": {"statistic": adf_stat, "p_value": adf_p, "stationary": adf_p < 0.05},
        "KPSS": {"statistic": kpss_stat, "p_value": kpss_p, "stationary": kpss_p > 0.05},
    }


def train_test_split_series(s: pd.Series, train_ratio: float = 0.8):
    n = int(len(s) * train_ratio)
    return s.iloc[:n], s.iloc[n:]


def fit_scaler(train: pd.Series) -> MinMaxScaler:
    return MinMaxScaler().fit(train.values.reshape(-1, 1))
