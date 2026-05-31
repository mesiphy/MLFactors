"""分层回测 — 按因子值分组计算收益。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LayeredResult:
    """分层回测结果。"""
    group_returns: pd.DataFrame          # 各组每期收益 (date × group)
    cumulative_returns: pd.DataFrame     # 各组累计收益
    annual_returns: pd.Series            # 各组年化收益
    sharpe_ratios: pd.Series             # 各组夏普比率
    long_max_drawdown: float = 0.0       # 多头最大回撤
    short_max_drawdown: float = 0.0      # 空头最大回撤
    top_max_drawdown: float = 0.0        # 兼容字段：多头最大回撤
    bottom_max_drawdown: float = 0.0     # 兼容字段：空头最大回撤
    top_excess_annual: float = 0.0       # 多头超额年化收益
    top_excess_max_drawdown: float = 0.0 # 多头超额最大回撤
    top_excess_calmar: float = 0.0       # 多头超额卡玛比率
    n_groups: int = 5


def _calc_max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    return float(drawdown.min()) if not drawdown.empty else 0.0


def layered_backtest(
    factor_df: pd.DataFrame | pd.Series,
    returns_df: pd.DataFrame | pd.Series,
    n_groups: int = 5,
    annual_trading_days: int = 252,
    period: int = 1,  # <==== 【新增参数】传入前向收益的周期天数
) -> LayeredResult:
    """按因子值分 N 组，计算各组收益表现。

    ``returns_df`` 应是每次调仓周期的收益。若输入是重叠 forward return，
    调用方应先按调仓频率抽样 factor/returns，否则 cumprod 会高估真实可交易收益。
    """
    if period <= 0:
        raise ValueError("period 必须为正整数")

    if isinstance(factor_df, pd.DataFrame):
        factor_df = factor_df.iloc[:, 0]
    if isinstance(returns_df, pd.DataFrame):
        returns_df = returns_df.iloc[:, 0]

    combined = pd.DataFrame({"factor": factor_df, "returns": returns_df}).dropna()
    if not isinstance(combined.index, pd.MultiIndex) or combined.index.nlevels < 2:
        raise ValueError("factor_df 和 returns_df 必须使用 MultiIndex: (date, asset)")

    # 每个截面日期分组
    dates = combined.index.get_level_values(0).unique().sort_values()
    group_ret_records: list[dict] = []

    for dt in dates:
        cross = combined.loc[dt].copy()
        if len(cross) < n_groups:
            continue
        # 先排名，避免离散因子大量并列值导致 qcut 无法稳定形成分组；
        # method="first" 会按原始顺序打破并列值。
        ranks = cross["factor"].rank(method="first")
        cross["group"] = pd.qcut(ranks, n_groups, labels=False, duplicates="drop") + 1
        for g in range(1, n_groups + 1):
            g_mask = cross["group"] == g
            if g_mask.sum() == 0:
                continue
            group_ret_records.append({
                "date": dt,
                "group": g,
                "returns": cross.loc[g_mask, "returns"].mean(),
            })

    if not group_ret_records:
        return LayeredResult(
            group_returns=pd.DataFrame(),
            cumulative_returns=pd.DataFrame(),
            annual_returns=pd.Series(dtype=float),
            sharpe_ratios=pd.Series(dtype=float),
            n_groups=n_groups,
        )

    ret_df = pd.DataFrame(group_ret_records)
    group_returns = ret_df.pivot(index="date", columns="group", values="returns").sort_index()
    periods_per_year = annual_trading_days / period
    
    # 累计收益
    cumulative = (1 + group_returns).cumprod() - 1

    # 年化收益
    total_ret = (1 + group_returns).prod(skipna=True)
    valid_periods = group_returns.notna().sum()
    annual_ret = total_ret.where(total_ret > 0) ** (
            periods_per_year / valid_periods.clip(lower=1)
        ) - 1
    annual_ret.name = "annual_return"

    # 夏普比率
    group_std = group_returns.std(ddof=1)
    sharpe = group_returns.mean().divide(group_std.where(group_std > 0)) * np.sqrt(periods_per_year)
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sharpe.name = "sharpe_ratio"

    top_group = int(group_returns.columns.max())
    bottom_group = int(group_returns.columns.min())
    long_max_drawdown = _calc_max_drawdown(group_returns[top_group])
    short_max_drawdown = _calc_max_drawdown(-group_returns[bottom_group])

    # 多头相对全市场的超额年化 (Top Group Excess Return)
    benchmark_returns = group_returns.mean(axis=1)
    top_excess_returns = (group_returns[top_group] - benchmark_returns).dropna()

    if top_excess_returns.empty:
        top_excess_annual = 0.0
        top_excess_max_drawdown = 0.0
    else:
        top_excess_total = (1 + top_excess_returns).prod(skipna=True)
        top_excess_valid_periods = max(int(top_excess_returns.notna().sum()), 1)
        top_excess_annual = (
            float(top_excess_total ** (periods_per_year / top_excess_valid_periods) - 1)
            if top_excess_total > 0
            else np.nan
        )

        # 多头超额最大回撤及卡玛比率
        top_excess_wealth = (1 + top_excess_returns).cumprod()
        top_excess_peak = top_excess_wealth.cummax()
        top_excess_drawdown = (top_excess_wealth - top_excess_peak) / top_excess_peak
        top_excess_max_drawdown = float(top_excess_drawdown.min()) if not top_excess_drawdown.empty else 0.0

    if np.isfinite(top_excess_annual) and top_excess_max_drawdown < 0:
        top_excess_calmar = float(top_excess_annual / abs(top_excess_max_drawdown))
    else:
        top_excess_calmar = 0.0
    
    return LayeredResult(
        group_returns=group_returns,
        cumulative_returns=cumulative,
        annual_returns=annual_ret,
        sharpe_ratios=sharpe,
        long_max_drawdown=long_max_drawdown,
        short_max_drawdown=short_max_drawdown,
        top_max_drawdown=long_max_drawdown,
        bottom_max_drawdown=short_max_drawdown,
        top_excess_annual=top_excess_annual,
        top_excess_max_drawdown=top_excess_max_drawdown,
        top_excess_calmar=top_excess_calmar,
        n_groups=n_groups,
    )
