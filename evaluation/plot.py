"""因子评估可视化。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from evaluation.selection.layered import LayeredResult


def plot_ic_series(
    ic_series: pd.Series,
    rolling_window: int = 20,
    title: str = "IC Time Series",
    figsize: tuple = (14, 5),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """IC 时间序列图 + 滚动均线。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.bar(ic_series.index, ic_series.values, width=0.8, alpha=0.4, color="steelblue", label="IC")
    rolling_mean = ic_series.rolling(rolling_window).mean()
    ax.plot(rolling_mean.index, rolling_mean.values, color="red", linewidth=1.5,
            label=f"MA({rolling_window})")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("IC")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_ic_histogram(
    ic_series: pd.Series,
    bins: int = 50,
    title: str = "IC Distribution",
    figsize: tuple = (8, 5),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """IC 分布直方图。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ic_clean = ic_series.dropna()
    ax.hist(ic_clean, bins=bins, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(ic_clean.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean={ic_clean.mean():.4f}")
    ax.set_title(title)
    ax.set_xlabel("IC")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_layered_returns(
    result: LayeredResult,
    benchmark_cumulative: pd.DataFrame | None = None,
    title: str = "Layered Cumulative Returns",
    figsize: tuple = (14, 6),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """分组累计收益曲线，可附加基准累计收益曲线。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    cum = result.cumulative_returns
    cmap = plt.cm.RdYlGn(np.linspace(0.1, 0.9, result.n_groups))
    for i, col in enumerate(cum.columns):
        ax.plot(cum.index, cum[col], label=f"Group {col}", color=cmap[i], linewidth=1.2)

    if benchmark_cumulative is not None and not benchmark_cumulative.empty:
        benchmark_colors = {"SPY": "black", "QQQ": "dimgray"}
        for col in benchmark_cumulative.columns:
            ax.plot(
                benchmark_cumulative.index,
                benchmark_cumulative[col],
                label=f"{col} Benchmark",
                color=benchmark_colors.get(str(col), "black"),
                linewidth=2,
                linestyle="--",
            )
    else:
        ax.plot(result.long_short_cumulative.index, result.long_short_cumulative.values,
                label="Long-Short", color="black", linewidth=2, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_ic_decay(
    decay_series: pd.Series,
    title: str = "IC Decay",
    figsize: tuple = (10, 5),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """IC 衰减图。"""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.bar(decay_series.index, decay_series.values, alpha=0.7, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Lag (periods)")
    ax.set_ylabel("Mean IC")
    ax.set_xticks(decay_series.index)
    fig.tight_layout()
    return fig


def plot_factor_report(
    ic_series: pd.Series,
    layered_result: LayeredResult,
    decay_series: pd.Series | None = None,
    benchmark_cumulative: pd.DataFrame | None = None,
    factor_name: str = "Factor",
    period: int | None = None,
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """综合报告图：4 合 1。"""
    n_rows = 2 if decay_series is None else 2
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    period_str = f" | Forward Period: {period}d" if period is not None else ""
    fig.suptitle(f"Factor Report: {factor_name}{period_str}", fontsize=14, fontweight="bold")

    plot_ic_series(ic_series, title=f"IC Time Series (period={period}d)", ax=axes[0, 0])
    plot_ic_histogram(ic_series, title=f"IC Distribution (period={period}d)", ax=axes[0, 1])
    plot_layered_returns(
        layered_result,
        benchmark_cumulative=benchmark_cumulative,
        title=f"Layered Returns (period={period}d)",
        ax=axes[1, 0],
    )

    if decay_series is not None:
        plot_ic_decay(decay_series, title="IC Decay", ax=axes[1, 1])
    else:
        axes[1, 1].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
