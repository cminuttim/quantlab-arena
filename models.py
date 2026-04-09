"""
models.py - All forecasting model wrappers with a common BaseModel interface.
"""

import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from features import build_feature_matrix, recursive_predict_ml, recursive_predict_torch

# Suppress sklearn/LightGBM feature-name compatibility warning (benign: predictions are correct)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseModel(ABC):
    """
    Common interface for all forecasting models.

    Protocol:
      1. fit(train_series)         — called once with data up to cutoff_date
      2. predict(context, horizon) — called at each forecast origin
    """

    name: str = "BaseModel"

    def __init__(self, horizon: int = 10, n_lags: int = 20):
        self.horizon = horizon
        self.n_lags = n_lags

    @abstractmethod
    def fit(self, train_series: pd.Series) -> None:
        """Train on all data up to cutoff_date."""
        ...

    @abstractmethod
    def predict(self, context: pd.Series, horizon: int) -> np.ndarray:
        """
        Generate point forecasts.

        Args:
            context: Full history up to (and including) origin date.
            horizon: Number of steps ahead to forecast.

        Returns:
            np.ndarray of shape (horizon,)
        """
        ...


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

class ArimaModel(BaseModel):
    name = "ARIMA"

    def fit(self, train_series: pd.Series) -> None:
        try:
            from pmdarima import auto_arima
            self.model = auto_arima(
                train_series.values,
                max_p=5, max_q=5, d=None,
                seasonal=False, stepwise=True,
                suppress_warnings=True, error_action="ignore",
                information_criterion="aic",
            )
        except Exception as e:
            print(f"  auto_arima failed ({e}), falling back to ARIMA(1,1,1)")
            from pmdarima import ARIMA
            self.model = ARIMA(order=(1, 1, 1))
            self.model.fit(train_series.values)
        self._last_update_date = train_series.index[-1]

    def predict(self, context: pd.Series, horizon: int) -> np.ndarray:
        new_obs = context[context.index > self._last_update_date]
        if len(new_obs) > 0:
            self.model.update(new_obs.values)
            self._last_update_date = context.index[-1]
        return self.model.predict(n_periods=horizon)


# ---------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------

class LinRegModel(BaseModel):
    name = "LinearRegression"

    def fit(self, train_series: pd.Series) -> None:
        from sklearn.linear_model import LinearRegression
        df = train_series.to_frame("Close")
        X, y = build_feature_matrix(df, self.n_lags)
        self._model = LinearRegression()
        self._model.fit(X.values, y.values)

    def predict(self, context: pd.Series, horizon: int) -> np.ndarray:
        return recursive_predict_ml(self._model, context, horizon, self.n_lags)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LGBMModel(BaseModel):
    name = "LightGBM"

    def fit(self, train_series: pd.Series) -> None:
        from lightgbm import LGBMRegressor
        df = train_series.to_frame("Close")
        X, y = build_feature_matrix(df, self.n_lags)
        self._model = LGBMRegressor(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )
        self._model.fit(X.values, y.values)

    def predict(self, context: pd.Series, horizon: int) -> np.ndarray:
        return recursive_predict_ml(self._model, context, horizon, self.n_lags)


# ---------------------------------------------------------------------------
# MLP (sklearn)
# ---------------------------------------------------------------------------

class MLPModel(BaseModel):
    name = "MLP"

    def fit(self, train_series: pd.Series) -> None:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        df = train_series.to_frame("Close")
        X, y = build_feature_matrix(df, self.n_lags)
        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_scaled = self._scaler_X.fit_transform(X.values)
        y_scaled = self._scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
        self._model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        self._model.fit(X_scaled, y_scaled)

    def predict(self, context: pd.Series, horizon: int) -> np.ndarray:
        # Wrap sklearn model with scalers for recursive prediction
        return recursive_predict_ml(_ScaledModel(self._model, self._scaler_X, self._scaler_y),
                                    context, horizon, self.n_lags)


class _ScaledModel:
    """Thin wrapper that applies StandardScaler around an sklearn estimator."""

    def __init__(self, model, scaler_X, scaler_y):
        self._model = model
        self._scaler_X = scaler_X
        self._scaler_y = scaler_y

    def predict(self, X):
        X_scaled = self._scaler_X.transform(X)
        y_scaled = self._model.predict(X_scaled)
        return self._scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


# ---------------------------------------------------------------------------
# CNN 1D (PyTorch)
# ---------------------------------------------------------------------------

class CNN1DModel(BaseModel):
    name = "CNN1D"

    def fit(self, train_series: pd.Series) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from features import build_sequence_dataset

        torch.manual_seed(42)
        self._device = torch.device("cpu")

        values = train_series.values.astype(np.float32)
        X, y = build_sequence_dataset(values, self.n_lags)

        # Normalize per-window
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True) + 1e-8
        X_norm = (X - means) / stds
        # Normalize targets relative to their input window
        y_norm = (y - means[:, 0, 0]) / stds[:, 0, 0]

        X_t = torch.tensor(X_norm)
        y_t = torch.tensor(y_norm)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self._net = _CNN1D(seq_len=self.n_lags).to(self._device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        self._net.train()
        for epoch in range(50):
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                pred = self._net(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

    def predict(self, context: pd.Series, horizon: int) -> np.ndarray:
        return recursive_predict_torch(
            self._net, context, horizon, self.n_lags, self._device
        )


class _CNN1D(object if True else None):
    pass


# Override _CNN1D with proper PyTorch module
try:
    import torch.nn as nn

    class _CNN1D(nn.Module):
        def __init__(self, seq_len: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            # x: (batch, seq_len, 1) → permute to (batch, 1, seq_len) for Conv1d
            x = x.permute(0, 2, 1)
            x = self.conv(x)
            return self.head(x)

except ImportError:
    pass


# ---------------------------------------------------------------------------
# GRU (PyTorch)
# ---------------------------------------------------------------------------

class GRUModel(BaseModel):
    name = "GRU"

    def fit(self, train_series: pd.Series) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from features import build_sequence_dataset

        torch.manual_seed(42)
        self._device = torch.device("cpu")

        values = train_series.values.astype(np.float32)
        X, y = build_sequence_dataset(values, self.n_lags)

        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True) + 1e-8
        X_norm = (X - means) / stds
        y_norm = (y - means[:, 0, 0]) / stds[:, 0, 0]

        X_t = torch.tensor(X_norm)
        y_t = torch.tensor(y_norm)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self._net = _GRUNet().to(self._device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        self._net.train()
        for epoch in range(50):
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                pred = self._net(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

    def predict(self, context: pd.Series, horizon: int) -> np.ndarray:
        return recursive_predict_torch(
            self._net, context, horizon, self.n_lags, self._device
        )


try:
    import torch.nn as nn

    class _GRUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_size=1, hidden_size=64, num_layers=2,
                              batch_first=True, dropout=0.1)
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            # x: (batch, seq_len, 1)
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])  # last timestep

except ImportError:
    pass


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

class ProphetModel(BaseModel):
    name = "Prophet"

    def fit(self, train_series: pd.Series) -> None:
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        # Store config for refitting at each origin
        self._prophet_kwargs = dict(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            # Higher changepoint_prior_scale makes the trend more flexible so it
            # tracks the current price level rather than anchoring to a long-run average.
            # Default (0.05) is far too stiff for volatile financial series.
            changepoint_prior_scale=0.5,
        )

    def _fit_on_context(self, context: pd.Series, lookback_years: int = 2):
        """
        Fits a fresh Prophet model on recent data up to the origin.
        Uses only the last `lookback_years` to keep the trend relevant to the
        current price level. 2 years is enough to capture yearly seasonality while
        avoiding distortion from older trend levels.
        """
        from prophet import Prophet
        cutoff = context.index[-1] - pd.DateOffset(years=lookback_years)
        recent = context[context.index >= cutoff]
        df_prophet = pd.DataFrame({"ds": recent.index, "y": recent.values})
        model = Prophet(**self._prophet_kwargs)
        model.fit(df_prophet)
        return model

    def predict(self, context: pd.Series, horizon: int) -> np.ndarray:
        # Refit on full context so the model incorporates post-cutoff observations.
        # This is equivalent to what ARIMA does via update() and ML models do via
        # the rolling lag window — without it, Prophet extrapolates from 2021 data
        # and is blind to the actual price level at the forecast origin.
        model = self._fit_on_context(context)
        origin = context.index[-1]
        future_dates = pd.bdate_range(start=origin, periods=horizon + 1, freq="B")[1:]
        future_df = pd.DataFrame({"ds": future_dates})
        forecast = model.predict(future_df)
        return forecast["yhat"].values[:horizon]


# ---------------------------------------------------------------------------
# Chronos (Amazon, pretrained zero-shot)
# ---------------------------------------------------------------------------

class ChronosModel(BaseModel):
    name = "Chronos"

    def fit(self, train_series: pd.Series) -> None:
        import torch
        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError(
                "chronos-forecasting not installed. Run: pip install chronos-forecasting"
            )

        print("  Loading Chronos model (downloads ~600MB on first run)...")
        self._pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    def predict(self, context: pd.Series, horizon: int) -> np.ndarray:
        import torch

        ctx = torch.tensor(context.values, dtype=torch.float32)
        samples = self._pipeline.predict(
            inputs=[ctx],
            prediction_length=horizon,
            num_samples=20,
            limit_prediction_length=False,
        )
        # samples: Tensor (1, num_samples, horizon)
        point_forecast = samples[0].median(dim=0).values.numpy()
        return point_forecast


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_all_models(horizon: int = 10, n_lags: int = 20, skip_chronos: bool = False) -> list[BaseModel]:
    """Returns list of all model instances."""
    models = [
        ArimaModel(horizon=horizon, n_lags=n_lags),
        LinRegModel(horizon=horizon, n_lags=n_lags),
        LGBMModel(horizon=horizon, n_lags=n_lags),
        MLPModel(horizon=horizon, n_lags=n_lags),
        CNN1DModel(horizon=horizon, n_lags=n_lags),
        GRUModel(horizon=horizon, n_lags=n_lags),
        ProphetModel(horizon=horizon, n_lags=n_lags),
    ]
    if not skip_chronos:
        models.append(ChronosModel(horizon=horizon, n_lags=n_lags))
    return models
