"""
features.py - Feature engineering for ML models.
"""

from collections import deque
import numpy as np
import pandas as pd


def make_lag_features(series: pd.Series, n_lags: int) -> pd.DataFrame:
    """Returns DataFrame with columns lag_1 ... lag_{n_lags}."""
    df = pd.DataFrame(index=series.index)
    for k in range(1, n_lags + 1):
        df[f"lag_{k}"] = series.shift(k)
    return df


def make_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Returns day_of_week (0-4) and month (1-12) features."""
    return pd.DataFrame(
        {"day_of_week": index.dayofweek, "month": index.month},
        index=index,
    )


def build_feature_matrix(
    df: pd.DataFrame,
    n_lags: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Builds (X, y) for supervised ML training.
    X columns: [lag_1, ..., lag_n, day_of_week, month]
    Drops first n_lags rows (NaN from lagging).
    """
    series = df["Close"]
    lags = make_lag_features(series, n_lags)
    cal = make_calendar_features(df.index)
    X = pd.concat([lags, cal], axis=1).dropna()
    y = series.loc[X.index]
    return X, y


def build_sequence_dataset(
    series: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding window dataset for single-step CNN/GRU training.
    X shape: (N, seq_len, 1)
    y shape: (N,)
    """
    X_list, y_list = [], []
    for i in range(len(series) - seq_len):
        window = series[i : i + seq_len]
        target = series[i + seq_len]
        X_list.append(window)
        y_list.append(target)
    X = np.array(X_list, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def recursive_predict_ml(
    model,
    context: pd.Series,
    horizon: int,
    n_lags: int,
) -> np.ndarray:
    """
    Recursive multi-step forecasting for sklearn-compatible models.
    Uses a rolling buffer of predicted values to generate lag features.

    Feature order: [lag_1, ..., lag_n, day_of_week, month]
    This must match the order used during training (build_feature_matrix).
    """
    buffer = deque(context.values[-n_lags:], maxlen=n_lags)
    predictions = []
    last_date = context.index[-1]

    # Generate business day offsets for future dates (best-effort approximation)
    future_dates = pd.bdate_range(start=last_date, periods=horizon + 1, freq="B")[1:]

    for step in range(horizon):
        next_date = future_dates[step]
        lag_feats = list(reversed(list(buffer)))  # [lag_1, lag_2, ...]
        dow = next_date.dayofweek
        month = next_date.month
        X = np.array([lag_feats + [dow, month]], dtype=np.float32)
        y_hat = float(model.predict(X)[0])
        buffer.append(y_hat)
        predictions.append(y_hat)

    return np.array(predictions)


def recursive_predict_torch(
    model,
    context: pd.Series,
    horizon: int,
    n_lags: int,
    device,
) -> np.ndarray:
    """
    Recursive multi-step forecasting for PyTorch sequence models (CNN/GRU).
    Normalizes input window per-window to handle non-stationary prices.
    """
    import torch

    model.eval()
    buffer = deque(context.values[-n_lags:], maxlen=n_lags)
    predictions = []

    with torch.no_grad():
        for _ in range(horizon):
            window = np.array(list(buffer), dtype=np.float32)
            mean = window.mean()
            std = window.std() + 1e-8
            window_norm = (window - mean) / std

            x = torch.tensor(window_norm, dtype=torch.float32).view(1, n_lags, 1).to(device)
            y_norm = model(x).item()
            y_hat = y_norm * std + mean
            buffer.append(y_hat)
            predictions.append(y_hat)

    return np.array(predictions)
