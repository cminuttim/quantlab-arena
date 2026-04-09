"""
evaluate.py - Evaluation loop and metrics computation.
"""

import numpy as np
import pandas as pd

from data import split_train_test, get_forecast_origins
from models import BaseModel


def run_evaluation(
    df: pd.DataFrame,
    models: list[BaseModel],
    cutoff_date: str,
    n_samples: int,
    horizon: int,
    test_days: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main evaluation loop.

    1. Fits all models on training data (up to cutoff_date).
    2. For each of n_samples sequential forecast origins in the test year:
       - Passes full history up to that origin as context
       - Calls predict() on each model
       - Records predictions vs actuals
    3. Computes aggregate metrics.

    Returns:
        (metrics_df, predictions_df)
    """
    train_df, test_df = split_train_test(df, cutoff_date, test_days)
    print(f"\nTrain: {train_df.index[0].date()} → {train_df.index[-1].date()} ({len(train_df)} days)")
    print(f"Test:  {test_df.index[0].date()} → {test_df.index[-1].date()} ({len(test_df)} days)")

    # Fit all models
    for model in models:
        print(f"\nFitting {model.name}...")
        try:
            model.fit(train_df["Close"])
        except Exception as e:
            print(f"  ERROR fitting {model.name}: {e}")
            model._fit_failed = True
            continue
        model._fit_failed = False

    # Get forecast origins
    origins = get_forecast_origins(test_df, n_samples, horizon)
    print(f"\nEvaluating {len(origins)} forecast origins (horizon={horizon} days each)...")

    all_preds = []

    for i, origin in enumerate(origins):
        print(f"  Origin {i+1}/{len(origins)}: {origin.date()}", end="")

        # Full history context up to and including origin
        context = df.loc[df.index <= origin, "Close"]

        # Actual future values
        future_idx = df.loc[df.index > origin].index
        if len(future_idx) < horizon:
            print(f" → skipped (only {len(future_idx)} future days available)")
            continue
        future_dates = future_idx[:horizon]
        actuals = df.loc[future_dates, "Close"].values

        print(f" → forecasting {horizon} steps")

        for model in models:
            if getattr(model, "_fit_failed", False):
                continue
            try:
                preds = model.predict(context, horizon)
                preds = np.asarray(preds, dtype=float)
                if len(preds) != horizon:
                    print(f"    WARNING: {model.name} returned {len(preds)} predictions, expected {horizon}")
                    preds = np.pad(preds, (0, horizon - len(preds)), constant_values=np.nan)
                last_obs = float(context.iloc[-1])
                for step, (pred, actual, date) in enumerate(zip(preds, actuals, future_dates), start=1):
                    all_preds.append({
                        "origin_date": origin,
                        "step": step,
                        "forecast_date": date,
                        "model": model.name,
                        "predicted": pred,
                        "actual": actual,
                        "last_obs": last_obs,
                    })
            except Exception as e:
                print(f"    WARNING: {model.name} failed at origin {origin.date()}: {e}")

    if not all_preds:
        raise RuntimeError("No predictions were generated. Check model fitting and data.")

    preds_df = pd.DataFrame(all_preds)
    metrics_df = compute_metrics(preds_df)
    return metrics_df, preds_df


def compute_metrics(preds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates predictions over all origins and steps.
    Returns DataFrame with scale-independent and scale-free metrics:
      model | MAE | MAPE | RMSE | SMAPE | MASE | TheilU | DirAcc | n_predictions

    MASE, TheilU, DirAcc are scale-free (benchmark against naive/random-walk)
    and valid for cross-ticker comparison.
    """
    has_last_obs = "last_obs" in preds_df.columns
    rows = []
    for model_name, grp in preds_df.groupby("model"):
        actual = grp["actual"].values
        predicted = grp["predicted"].values
        row = {
            "model": model_name,
            "MAE": _mae(actual, predicted),
            "MAPE": _mape(actual, predicted),
            "RMSE": _rmse(actual, predicted),
            "SMAPE": _smape(actual, predicted),
        }
        if has_last_obs:
            last_obs = grp["last_obs"].values
            row["MASE"] = _mase(actual, predicted, last_obs)
            row["TheilU"] = _theil_u(actual, predicted, last_obs)
            row["DirAcc"] = _directional_accuracy(actual, predicted, last_obs)
        row["n_predictions"] = len(grp)
        rows.append(row)
    metrics_df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    return metrics_df


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.nanmean(np.abs(actual - predicted)))


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(actual == 0, np.nan, np.abs(actual - predicted) / np.abs(actual)) * 100
    return float(np.nanmean(pct))


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((actual - predicted) ** 2)))


def _smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denom = np.abs(actual) + np.abs(predicted)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where(denom == 0, np.nan, 2 * np.abs(actual - predicted) / denom) * 100
    return float(np.nanmean(val))


def _mase(actual: np.ndarray, predicted: np.ndarray, last_obs: np.ndarray) -> float:
    """MAE of model / MAE of naive (random walk). <1 beats naive."""
    mae_naive = float(np.nanmean(np.abs(actual - last_obs)))
    if mae_naive == 0:
        return np.nan
    return _mae(actual, predicted) / mae_naive


def _theil_u(actual: np.ndarray, predicted: np.ndarray, last_obs: np.ndarray) -> float:
    """RMSE of model / RMSE of naive (random walk). <1 beats naive."""
    rmse_naive = float(np.sqrt(np.nanmean((actual - last_obs) ** 2)))
    if rmse_naive == 0:
        return np.nan
    return _rmse(actual, predicted) / rmse_naive


def _directional_accuracy(actual: np.ndarray, predicted: np.ndarray, last_obs: np.ndarray) -> float:
    """
    % of steps where model predicts the correct direction vs last observed value.
    Excludes ties (actual == last_obs). 50% = random; 100% = perfect direction.
    """
    sign_actual = np.sign(actual - last_obs)
    sign_pred = np.sign(predicted - last_obs)
    valid = sign_actual != 0
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(sign_actual[valid] == sign_pred[valid])) * 100
