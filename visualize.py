"""
visualize.py - Plotting functions for model comparison results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Consistent color palette for models
_PALETTE = list(mcolors.TABLEAU_COLORS.values())


def _model_colors(model_names: list[str]) -> dict[str, str]:
    return {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(sorted(model_names))}


def _add_param_footer(fig, params: dict) -> None:
    """
    Adds a compact parameter summary as a small footer at the bottom of a figure.
    params keys: ticker, train_start, cutoff_date, test_days, horizon, lags, n_samples
    """
    parts = []
    if "train_start" in params and "cutoff_date" in params:
        parts.append(f"Train: {params['train_start']} → {params['cutoff_date']}")
    if "test_days" in params:
        parts.append(f"Test: {params['test_days']} days")
    if "horizon" in params:
        parts.append(f"Horizon: {params['horizon']} days")
    if "lags" in params:
        parts.append(f"Lags: {params['lags']}")
    if "n_samples" in params:
        parts.append(f"Samples: {params['n_samples']}")
    text = "  |  ".join(parts)
    fig.text(0.5, 0.005, text, ha="center", va="bottom",
             fontsize=7, color="#666666", style="italic")


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    output_path: str,
    ticker: str = "",
    params: dict | None = None,
) -> None:
    """
    2x2 figure with bar charts for MAE, MAPE, RMSE, SMAPE.
    Models sorted by MAE (best to worst).
    """
    metrics = ["MAE", "MAPE", "RMSE", "SMAPE"]
    labels = {
        "MAE": "MAE (index points)",
        "MAPE": "MAPE (%)",
        "RMSE": "RMSE (index points)",
        "SMAPE": "SMAPE (%)",
    }

    colors = _model_colors(metrics_df["model"].tolist())
    bar_colors = [colors[m] for m in metrics_df["model"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    title = f"Forecasting Model Comparison — {ticker}" if ticker else "Forecasting Model Comparison"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, metric in zip(axes.flat, metrics):
        df_sorted = metrics_df.sort_values(metric)
        bars = ax.bar(df_sorted["model"], df_sorted[metric],
                      color=[colors[m] for m in df_sorted["model"]])
        ax.set_title(metric, fontsize=11)
        ax.set_ylabel(labels[metric])
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        # Value labels on bars
        for bar, val in zip(bars, df_sorted[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    if params:
        _add_param_footer(fig, params)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics chart → {output_path}")


def plot_predictions_sample(
    preds_df: pd.DataFrame,
    origin_date: pd.Timestamp,
    sample_idx: int,
    full_series: pd.Series,
    output_path: str,
    context_days: int = 30,
    params: dict | None = None,
) -> None:
    """
    For one forecast origin: plots historical context + all model forecasts vs actual.
    """
    # Historical context (last context_days before origin)
    hist_start = full_series.index[max(0, full_series.index.get_loc(origin_date) - context_days)]
    history = full_series.loc[hist_start:origin_date]

    # Actual future
    origin_preds = preds_df[preds_df["origin_date"] == origin_date]
    if origin_preds.empty:
        return

    models = sorted(origin_preds["model"].unique())
    colors = _model_colors(models)

    # Actuals (same for all models)
    actual_sample = origin_preds[origin_preds["model"] == models[0]].sort_values("step")
    actual_dates = actual_sample["forecast_date"].values
    actual_vals = actual_sample["actual"].values

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(
        f"Forecast from {origin_date.date()}  |  Sample {sample_idx + 1}",
        fontsize=12,
    )

    # Historical
    ax.plot(history.index, history.values, color="gray", linewidth=1.5,
            label="History", zorder=1)
    # Vertical line at origin
    ax.axvline(origin_date, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    # Actual future
    ax.plot(actual_dates, actual_vals, color="black", linewidth=2.0,
            marker="o", markersize=4, label="Actual", zorder=2)

    # Model predictions
    for model_name in models:
        model_preds = origin_preds[origin_preds["model"] == model_name].sort_values("step")
        ax.plot(model_preds["forecast_date"].values, model_preds["predicted"].values,
                color=colors[model_name], linewidth=1.5, linestyle="--",
                marker=".", markersize=5, label=model_name, alpha=0.85, zorder=3)

    ax.set_xlabel("Date")
    ticker_label = params.get("ticker", "") if params else ""
    ax.set_ylabel(f"{ticker_label} Close" if ticker_label else "Close Price")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    if params:
        _add_param_footer(fig, params)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved sample plot → {output_path}")


def plot_error_distribution(
    preds_df: pd.DataFrame,
    output_path: str,
    ticker: str = "",
    params: dict | None = None,
) -> None:
    """
    Violin + boxplot of absolute error distributions per model.
    Models are ordered by median absolute error (best left).
    """
    preds_df = preds_df.copy()
    preds_df["abs_error"] = (preds_df["predicted"] - preds_df["actual"]).abs()

    # Order models by median absolute error
    order = (
        preds_df.groupby("model")["abs_error"]
        .median()
        .sort_values()
        .index.tolist()
    )
    colors = _model_colors(order)

    data_by_model = [preds_df[preds_df["model"] == m]["abs_error"].dropna().values for m in order]
    positions = range(1, len(order) + 1)

    fig, ax = plt.subplots(figsize=(max(10, len(order) * 1.4), 6))
    title = f"Absolute Error Distribution by Model"
    if ticker:
        title += f"  —  {ticker}"
    ax.set_title(title, fontsize=12, fontweight="bold")

    parts = ax.violinplot(data_by_model, positions=positions, showmedians=False,
                          showextrema=False)
    for i, (pc, model) in enumerate(zip(parts["bodies"], order)):
        pc.set_facecolor(colors[model])
        pc.set_alpha(0.6)

    # Overlay boxplot (no fliers — already shown by violin)
    bp = ax.boxplot(data_by_model, positions=positions, widths=0.15,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=2),
                    boxprops=dict(facecolor="white", alpha=0.8),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))

    ax.set_xticks(list(positions))
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("Absolute Error")
    ax.set_xlabel("Model  (ordered by median error)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(0.3, len(order) + 0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    if params:
        _add_param_footer(fig, params)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved error distribution → {output_path}")


def plot_signed_error(
    preds_df: pd.DataFrame,
    output_path: str,
    ticker: str = "",
    params: dict | None = None,
) -> None:
    """
    Two-panel figure for signed error analysis (predicted - actual):
      Top:    Violin + box of signed errors per model, with 0-line.
              Positive → over-prediction; negative → under-prediction.
      Bottom: Mean signed error bar chart — quick bias summary per model.
    Models ordered by mean signed error (most negative to most positive).
    """
    preds_df = preds_df.copy()
    preds_df["signed_error"] = preds_df["predicted"] - preds_df["actual"]

    order = (
        preds_df.groupby("model")["signed_error"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    colors = _model_colors(order)
    data_by_model = [preds_df[preds_df["model"] == m]["signed_error"].dropna().values for m in order]
    positions = list(range(1, len(order) + 1))
    means = [d.mean() for d in data_by_model]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(10, len(order) * 1.4), 10),
                                         gridspec_kw={"height_ratios": [2, 1]})
    suptitle = "Signed Error Analysis  (predicted − actual)"
    if ticker:
        suptitle += f"  —  {ticker}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    # --- Top panel: violin + box ---
    parts = ax_top.violinplot(data_by_model, positions=positions, showmedians=False,
                              showextrema=False)
    for pc, model in zip(parts["bodies"], order):
        pc.set_facecolor(colors[model])
        pc.set_alpha(0.55)

    ax_top.boxplot(data_by_model, positions=positions, widths=0.15,
                   patch_artist=True, showfliers=False,
                   medianprops=dict(color="black", linewidth=2),
                   boxprops=dict(facecolor="white", alpha=0.8),
                   whiskerprops=dict(linewidth=1.2),
                   capprops=dict(linewidth=1.2))

    ax_top.axhline(0, color="red", linewidth=1.2, linestyle="--", alpha=0.8, label="No bias (0)")
    ax_top.set_xticks(positions)
    ax_top.set_xticklabels(order, rotation=25, ha="right")
    ax_top.set_ylabel("Signed Error  (predicted − actual)")
    ax_top.set_xlim(0.3, len(order) + 0.7)
    ax_top.grid(axis="y", alpha=0.3)
    ax_top.legend(fontsize=9)

    # Annotate median per model
    for pos, d, model in zip(positions, data_by_model, order):
        med = float(np.median(d))
        ax_top.text(pos, med, f" {med:+.0f}", va="center", ha="left", fontsize=7,
                    color=colors[model], fontweight="bold")

    # --- Bottom panel: mean bias bar chart ---
    bar_colors = [
        "#d62728" if m > 0 else "#1f77b4"   # red = over-predict, blue = under-predict
        for m in means
    ]
    bars = ax_bot.bar(order, means, color=bar_colors, alpha=0.8, edgecolor="white")
    ax_bot.axhline(0, color="black", linewidth=0.8)
    ax_bot.set_ylabel("Mean Signed Error")
    ax_bot.set_xlabel("Model  (ordered by mean bias, negative → under-prediction)")
    ax_bot.tick_params(axis="x", rotation=25)
    ax_bot.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, means):
        ypos = val + (ax_bot.get_ylim()[1] - ax_bot.get_ylim()[0]) * 0.01 if val >= 0 else val - (ax_bot.get_ylim()[1] - ax_bot.get_ylim()[0]) * 0.04
        ax_bot.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:+.0f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    if params:
        _add_param_footer(fig, params)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved signed error chart → {output_path}")


def plot_error_ci(
    preds_df: pd.DataFrame,
    output_path: str,
    ticker: str = "",
    params: dict | None = None,
    ci: float = 0.95,
    method: str = "bootstrap",
    n_bootstrap: int = 10000,
) -> None:
    """
    Forest plot of per-model absolute error confidence intervals.

    method="bootstrap" (default):
        Percentile bootstrap CI for the median absolute error.
        Resamples errors n_bootstrap times; the dot shows the sample median.
        Interpretation: uncertainty about the typical (median) error level.

    method="gamma":
        Parametric prediction interval from a fitted Gamma distribution (loc=0).
        The dot shows the distribution mean.
        Interpretation: range within which individual errors are expected to fall.

    Models ordered by median absolute error (best at top).
    """
    from scipy import stats as st

    preds_df = preds_df.copy()
    preds_df["abs_error"] = (preds_df["predicted"] - preds_df["actual"]).abs()

    order = (
        preds_df.groupby("model")["abs_error"]
        .median()
        .sort_values()
        .index.tolist()
    )
    colors = _model_colors(order)

    alpha = 1 - ci
    rows = []
    for model in order:
        errors = preds_df[preds_df["model"] == model]["abs_error"].dropna().values

        if method == "bootstrap":
            rng = np.random.default_rng(1)
            boot_medians = np.array([
                np.median(rng.choice(errors, size=len(errors), replace=True))
                for _ in range(n_bootstrap)
            ])
            ci_low  = float(np.percentile(boot_medians, alpha / 2 * 100))
            ci_high = float(np.percentile(boot_medians, (1 - alpha / 2) * 100))
            center  = float(np.median(errors))
            note    = f"bootstrap n={n_bootstrap:,}  |  median={center:.1f}"
        else:  # gamma
            try:
                shape, loc, scale = st.gamma.fit(errors, floc=0)
                dist   = st.gamma(shape, loc=loc, scale=scale)
                ci_low  = dist.ppf(alpha / 2)
                ci_high = dist.ppf(1 - alpha / 2)
                center  = dist.mean()
                _, ks_p = st.kstest(errors, "gamma", args=(shape, loc, scale))
                fit_tag = "good fit" if ks_p >= 0.05 else "poor fit"
                note = (
                    f"Γ(shape={shape:.2f}, scale={scale:.2f})  |  "
                    f"KS p={ks_p:.3f} ({fit_tag})"
                )
            except Exception:
                ci_low  = float(np.percentile(errors, alpha / 2 * 100))
                ci_high = float(np.percentile(errors, (1 - alpha / 2) * 100))
                center  = float(np.mean(errors))
                note    = "empirical percentiles (Gamma fit failed)"

        rows.append(dict(model=model, center=center, ci_low=ci_low,
                         ci_high=ci_high, note=note))

    method_label = (
        f"Bootstrap CI for median  ({n_bootstrap:,} resamples)"
        if method == "bootstrap"
        else "Gamma prediction interval"
    )
    n = len(order)
    fig, ax = plt.subplots(figsize=(12, max(5, n * 1.1 + 2)))
    title = f"Absolute Error — {int(ci * 100)}% Confidence Intervals ({method_label})"
    if ticker:
        title += f"  —  {ticker}"
    ax.set_title(title, fontsize=12, fontweight="bold")

    for i, row in enumerate(rows):
        color = colors[row["model"]]
        # Horizontal CI bar
        ax.plot([row["ci_low"], row["ci_high"]], [i, i],
                color=color, linewidth=2.5, solid_capstyle="round", zorder=3)
        # End caps
        for xv in (row["ci_low"], row["ci_high"]):
            ax.plot([xv, xv], [i - 0.18, i + 0.18], color=color, linewidth=2, zorder=3)
        # Center dot
        ax.scatter([row["center"]], [i], color=color, s=70, zorder=5)
        # Method info below the bar
        ax.text(row["ci_low"], i - 0.28, row["note"],
                va="top", ha="left", fontsize=7, color="#555555", style="italic")

    ax.set_yticks(range(n))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Model  (ordered by median error)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_ylim(-0.9, n - 0.3)

    center_label = "Sample median" if method == "bootstrap" else "Distribution mean"
    ci_label     = f"{int(ci * 100)}% CI for median" if method == "bootstrap" else f"{int(ci * 100)}% prediction interval"
    ax.scatter([], [], color="gray", s=70, label=center_label)
    ax.plot([], [], color="gray", linewidth=2.5, label=ci_label)
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    if params:
        _add_param_footer(fig, params)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved CI plot → {output_path}")


def plot_ticker_comparison(
    all_metrics_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    2x2 heatmap grid comparing models across tickers for scale-free metrics.
    Rows = tickers, columns = models.

    Metrics shown:
      MAPE    — lower is better (YlOrRd colormap)
      MASE    — <1 beats naive; diverging colormap centered on 1
      TheilU  — <1 beats naive; diverging colormap centered on 1
      DirAcc  — higher is better (YlGn colormap)
    """
    import matplotlib.colors as mcolors

    metrics_cfg = [
        ("MAPE",   "MAPE (%)",               "YlOrRd",   "lower"),
        ("MASE",   "MASE  (< 1 = beats naive)", "RdYlGn_r", "diverge1"),
        ("TheilU", "Theil U  (< 1 = beats naive)", "RdYlGn_r", "diverge1"),
        ("DirAcc", "Directional Accuracy (%)", "YlGn",     "higher"),
    ]

    # Drop metrics not present (e.g., old CSVs)
    metrics_cfg = [(m, l, c, b) for m, l, c, b in metrics_cfg if m in all_metrics_df.columns]
    if not metrics_cfg:
        print("  WARNING: no scale-free metrics found in combined data, skipping heatmap")
        return

    n = len(metrics_cfg)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    axes = np.array(axes).reshape(nrows, ncols)

    fig.suptitle("Cross-Ticker Model Comparison — Scale-Free Metrics",
                 fontsize=14, fontweight="bold")

    for idx, (metric, label, cmap, better) in enumerate(metrics_cfg):
        ax = axes[idx // ncols, idx % ncols]
        pivot = (
            all_metrics_df.pivot_table(index="ticker", columns="model", values=metric, aggfunc="mean")
            .sort_index()
        )
        pivot = pivot[sorted(pivot.columns)]  # sort models alphabetically

        vals = pivot.values.astype(float)
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))

        if better == "diverge1" and vmin < 1.0 < vmax:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.75)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)

        # Annotate cells
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                val = vals[r, c]
                if not np.isnan(val):
                    fmt = f"{val:.1f}%" if metric == "DirAcc" else f"{val:.3f}"
                    ax.text(c, r, fmt, ha="center", va="center", fontsize=8)

    # Hide unused axes if odd number of metrics
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ticker comparison heatmap → {output_path}")


def print_metrics_table(metrics_df: pd.DataFrame) -> None:
    """Pretty-prints the metrics table to stdout."""
    has_new = "MASE" in metrics_df.columns
    if has_new:
        width = 93
        print("\n" + "=" * width)
        print("MODEL COMPARISON RESULTS")
        print("=" * width)
        print(f"{'Model':<20} {'MAE':>10} {'MAPE%':>8} {'RMSE':>10} {'SMAPE%':>8} {'MASE':>7} {'TheilU':>8} {'DirAcc%':>9}")
        print("-" * width)
        for _, row in metrics_df.iterrows():
            print(
                f"{row['model']:<20} {row['MAE']:>10.2f} {row['MAPE']:>8.2f} "
                f"{row['RMSE']:>10.2f} {row['SMAPE']:>8.2f} "
                f"{row['MASE']:>7.3f} {row['TheilU']:>8.3f} {row['DirAcc']:>8.1f}%"
            )
        print("=" * width)
    else:
        width = 65
        print("\n" + "=" * width)
        print("MODEL COMPARISON RESULTS")
        print("=" * width)
        print(f"{'Model':<20} {'MAE':>10} {'MAPE%':>10} {'RMSE':>10} {'SMAPE%':>10}")
        print("-" * width)
        for _, row in metrics_df.iterrows():
            print(
                f"{row['model']:<20} {row['MAE']:>10.2f} {row['MAPE']:>10.2f} "
                f"{row['RMSE']:>10.2f} {row['SMAPE']:>10.2f}"
            )
        print("=" * width)
