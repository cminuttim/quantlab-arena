"""
main.py - CLI entry point for financial time series forecasting model comparison.

Usage:
    python main.py --cutoff-date 2022-01-01 --n-samples 12 --horizon 10
    python main.py --cutoff-date 2022-01-01 --n-samples 3 --skip-chronos   # quick test
    python main.py --ticker SPY --start-date 2010-01-01 --cutoff-date 2023-01-01
    python main.py --ticker ^GSPC --cutoff-date 2022-01-01 --n-samples 6
    python main.py --help
"""

import argparse
import os
import sys
import numpy as np

# Reproducibility
np.random.seed(42)
try:
    import torch
    torch.manual_seed(42)
except ImportError:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QuantLab Arena — financial time series forecasting benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help=(
            "Start date for historical data (YYYY-MM-DD). "
            "Defaults to the earliest available data for the ticker. "
            "Useful to limit training history or reduce download size."
        ),
    )
    parser.add_argument(
        "--cutoff-date",
        default="2022-01-01",
        help="Training cutoff date (YYYY-MM-DD). All data before this date is used for training.",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=252,
        help="Number of trading days in the test period after cutoff-date. Default: 252 (~1 year).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=12,
        help="Number of sequential forecast origins to evaluate within the test period.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Forecast horizon in trading days (10 ≈ 2 weeks).",
    )
    parser.add_argument(
        "--lags",
        type=int,
        default=20,
        help="Number of lag features for ML/DL models.",
    )
    parser.add_argument(
        "--ticker",
        default="^MXX",
        help=(
            "Yahoo Finance ticker symbol. Examples: ^MXX (IPC Mexico), "
            "^GSPC (S&P 500), SPY, AAPL, BTC-USD, EURUSD=X."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="Directory to cache downloaded data.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to write output CSV files and plots.",
    )
    parser.add_argument(
        "--skip-chronos",
        action="store_true",
        help="Skip the Chronos model (useful for offline runs or faster testing).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Subset of models to run. Options: arima linreg lgbm mlp cnn gru prophet chronos. "
            "Runs all models if not specified."
        ),
    )
    parser.add_argument(
        "--ci-method",
        default="bootstrap",
        choices=["bootstrap", "gamma"],
        help=(
            "Method for error_ci.png: 'bootstrap' (CI for the median, default) or "
            "'gamma' (parametric prediction interval from a fitted Gamma distribution)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("QuantLab Arena — Financial Time Series Forecasting Benchmark")
    print("=" * 60)
    print(f"Ticker:       {args.ticker}")
    print(f"Start date:   {args.start_date or 'earliest available'}")
    print(f"Cutoff date:  {args.cutoff_date}")
    print(f"Test days:    {args.test_days} trading days")
    print(f"Test samples: {args.n_samples}")
    print(f"Horizon:      {args.horizon} trading days (~2 weeks)")
    print(f"Lags:         {args.lags}")
    print(f"Skip Chronos: {args.skip_chronos}")
    print()

    # Create output directories — organized per ticker
    ticker_slug = args.ticker.replace("^", "").replace("/", "-")
    ticker_dir = os.path.join(args.output_dir, ticker_slug)
    plots_dir = os.path.join(ticker_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Load data
    from data import load_data, DEFAULT_START
    df = load_data(
        ticker=args.ticker,
        start_date=args.start_date or DEFAULT_START,
        cutoff_date=args.cutoff_date,
        test_days=args.test_days,
        horizon=args.horizon,
        cache_dir=args.cache_dir,
    )
    print(f"Data loaded: {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})")

    # 2. Build models
    from models import get_all_models, ArimaModel, LinRegModel, LGBMModel, MLPModel
    from models import CNN1DModel, GRUModel, ProphetModel, ChronosModel

    _MODEL_MAP = {
        "arima": ArimaModel,
        "linreg": LinRegModel,
        "lgbm": LGBMModel,
        "mlp": MLPModel,
        "cnn": CNN1DModel,
        "gru": GRUModel,
        "prophet": ProphetModel,
        "chronos": ChronosModel,
    }

    if args.models:
        unknown = [m for m in args.models if m.lower() not in _MODEL_MAP]
        if unknown:
            print(f"ERROR: Unknown models: {unknown}. Valid options: {list(_MODEL_MAP)}")
            sys.exit(1)
        skip_chronos = "chronos" not in [m.lower() for m in args.models]
        models = [
            _MODEL_MAP[m.lower()](horizon=args.horizon, n_lags=args.lags)
            for m in args.models
        ]
    else:
        models = get_all_models(
            horizon=args.horizon,
            n_lags=args.lags,
            skip_chronos=args.skip_chronos,
        )

    print(f"\nModels to evaluate: {[m.name for m in models]}")

    # 3. Run evaluation
    from evaluate import run_evaluation
    metrics_df, preds_df = run_evaluation(
        df=df,
        models=models,
        cutoff_date=args.cutoff_date,
        n_samples=args.n_samples,
        horizon=args.horizon,
        test_days=args.test_days,
    )

    # 4. Save results — add ticker column and write to per-ticker subdir
    import pandas as pd
    metrics_df.insert(0, "ticker", args.ticker)
    preds_df.insert(0, "ticker", args.ticker)
    metrics_path = os.path.join(ticker_dir, "metrics_summary.csv")
    preds_path = os.path.join(ticker_dir, "predictions.csv")
    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)
    print(f"\nSaved metrics → {metrics_path}")
    print(f"Saved predictions → {preds_path}")

    # 5. Print metrics table
    from visualize import print_metrics_table
    print_metrics_table(metrics_df)

    # 6. Generate plots
    from visualize import (plot_metrics_comparison, plot_predictions_sample,
                           plot_error_distribution, plot_signed_error,
                           plot_error_ci)

    plot_params = {
        "ticker":      args.ticker,
        "train_start": str(df.index[0].date()),
        "cutoff_date": args.cutoff_date,
        "test_days":   args.test_days,
        "horizon":     args.horizon,
        "lags":        args.lags,
        "n_samples":   args.n_samples,
    }
    from data import get_forecast_origins, split_train_test

    metrics_plot = os.path.join(plots_dir, "comparison_metrics.png")
    plot_metrics_comparison(metrics_df, metrics_plot, ticker=args.ticker, params=plot_params)

    plot_error_distribution(
        preds_df,
        os.path.join(plots_dir, "error_distribution.png"),
        ticker=args.ticker,
        params=plot_params,
    )
    plot_signed_error(
        preds_df,
        os.path.join(plots_dir, "signed_error.png"),
        ticker=args.ticker,
        params=plot_params,
    )
    plot_error_ci(
        preds_df,
        os.path.join(plots_dir, "error_ci.png"),
        ticker=args.ticker,
        params=plot_params,
        method=args.ci_method,
    )

    _, test_df = split_train_test(df, args.cutoff_date, args.test_days)
    origins = get_forecast_origins(test_df, args.n_samples, args.horizon)
    full_series = df["Close"]

    for i, origin in enumerate(origins):
        sample_plot = os.path.join(plots_dir, f"predictions_sample_{i+1:02d}.png")
        plot_predictions_sample(
            preds_df=preds_df,
            origin_date=origin,
            sample_idx=i,
            full_series=full_series,
            output_path=sample_plot,
            params=plot_params,
        )

    # 7. Cross-ticker comparison (runs if ≥2 ticker subdirs exist)
    import pandas as pd
    all_frames = []
    for entry in os.scandir(args.output_dir):
        if not entry.is_dir():
            continue
        candidate = os.path.join(entry.path, "metrics_summary.csv")
        if os.path.exists(candidate):
            try:
                all_frames.append(pd.read_csv(candidate))
            except Exception:
                pass

    if len(all_frames) > 1:
        from visualize import plot_ticker_comparison
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = os.path.join(args.output_dir, "all_metrics.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nSaved combined metrics → {combined_path}")
        plot_ticker_comparison(
            combined,
            os.path.join(args.output_dir, "ticker_comparison_heatmap.png"),
        )

    print(f"\nAll done. Results saved to '{ticker_dir}/'")


if __name__ == "__main__":
    main()
