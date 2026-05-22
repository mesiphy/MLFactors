"""IC / RankIC / ICIR 及真实换手率等因子评估指标。"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats

# 忽略因截面因子值全部相同（常数）导致的 Spearman 警告
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)


def calc_ic(
    factor: pd.Series,
    returns: pd.Series,
    method: str = "rank",
) -> float:
    """计算单期截面 IC。

    Parameters
    ----------
    factor : 因子值 Series（同一截面的股票）
    returns : 同期前向收益 Series
    method : "rank" (Spearman) 或 "pearson"

    Returns
    -------
    IC 值 (float)
    """
    aligned = pd.DataFrame({"factor": factor, "returns": returns}).dropna()
    if len(aligned) < 3:
        return np.nan

    if method == "rank":
        corr, _ = stats.spearmanr(aligned["factor"], aligned["returns"])
    else:
        corr, _ = stats.pearsonr(aligned["factor"], aligned["returns"])
        
    return corr


def calc_ic_series(
    factor_df: pd.DataFrame | pd.Series,
    returns_df: pd.DataFrame | pd.Series,
    method: str = "rank",
) -> pd.Series:
    """逐期计算截面 IC，返回 IC 时间序列（已通过 groupby 向量化加速）。"""
    if isinstance(factor_df, pd.DataFrame):
        factor_df = factor_df.iloc[:, 0]
    if isinstance(returns_df, pd.DataFrame):
        returns_df = returns_df.iloc[:, 0]

    combined = pd.DataFrame({"factor": factor_df, "returns": returns_df}).dropna()

    # 核心优化：使用 groupby 替代 for 循环，速度提升显著
    def _cross_sectional_ic(df):
        if len(df) < 3:
            return np.nan
        if method == "rank":
            return stats.spearmanr(df["factor"], df["returns"])[0]
        else:
            return stats.pearsonr(df["factor"], df["returns"])[0]

    ic_series = combined.groupby(level=0).apply(_cross_sectional_ic)
    return pd.Series(ic_series, name="IC").sort_index()


def calc_icir(
    ic_series: pd.Series, 
    period: int = 1, 
    annualize: bool = False, 
    periods_per_year: int = 252
) -> float:
    """计算信息比率 ICIR，并自动处理多期重叠的统计惩罚。
    
    Parameters
    ----------
    ic_series : pd.Series，IC 时间序列
    period : int，前向收益的周期数（用于消除重叠自相关造成的虚高）
    annualize : bool，是否年化（默认 False）
    periods_per_year : int，一年包含的交易日数（日频默认 252）
    """
    ic_clean = ic_series.dropna()
    if len(ic_clean) < 2 or ic_clean.std() == 0:
        return np.nan
        
    # 1. 计算原始的、带有重叠虚高水分的基础 ICIR
    base_icir = ic_clean.mean() / ic_clean.std()
    
    # 2. 消除多期重叠导致的自相关虚高
    adjusted_icir = base_icir / np.sqrt(period)
    
    # 3. 年化处理
    if annualize:
        # 乘以 sqrt(periods_per_year) 完成年化。
        # 最终数学等价于：base_icir * sqrt(periods_per_year / period)
        return adjusted_icir * np.sqrt(periods_per_year)
    else:
        return adjusted_icir


def calc_ic_decay(
    factor_df: pd.DataFrame | pd.Series,
    returns_provider,
    max_lag: int = 20,
    method: str = "rank",
) -> pd.Series:
    """计算 IC 衰减曲线（修复了收益错位平移的 Bug）。"""
    if isinstance(factor_df, pd.DataFrame):
        factor_s = factor_df.iloc[:, 0]
    else:
        factor_s = factor_df

    if isinstance(returns_provider, (pd.DataFrame, pd.Series)):
        if isinstance(returns_provider, pd.DataFrame):
            ret_s = returns_provider.iloc[:, 0]
        else:
            ret_s = returns_provider

        ret_unstacked = ret_s.unstack()
        decay = {}
        for lag in range(1, max_lag + 1):
            # 修正：当 lag=1 时不应平移（0），lag=2 时平移 1 期
            shifted_ret = ret_unstacked.shift(-(lag - 1)).stack()
            ic_s = calc_ic_series(factor_s, shifted_ret, method)
            decay[lag] = ic_s.mean()
        return pd.Series(decay, name="IC_decay")

    # 如果 returns_provider 是可调用对象
    decay = {}
    for lag in range(1, max_lag + 1):
        ret = returns_provider(lag) if callable(returns_provider) else returns_provider[lag]
        ic_s = calc_ic_series(factor_s, ret, method)
        decay[lag] = ic_s.mean()
    return pd.Series(decay, name="IC_decay")


def calc_turnover(
    factor_df: pd.DataFrame | pd.Series, 
    quantiles: int = 5
) -> pd.Series:
    """计算真实资金组合换手率。
    
    逻辑：每期将股票按因子值分为 `quantiles` 组，构建做多头部（Top Quantile）的等权投资组合。
    单边换手率 = sum(|W_{t} - W_{t-1}|) / 2

    Parameters
    ----------
    factor_df : 因子值，MultiIndex(date, symbol)
    quantiles : 分组数量，默认 5 组（即取 Top 20% 建仓）

    Returns
    -------
    pd.Series，索引为 date，值为单边资金换手率
    """
    if isinstance(factor_df, pd.DataFrame):
        factor_s = factor_df.iloc[:, 0]
    else:
        factor_s = factor_df

    # 转为宽表 (date x symbol)
    factor_unstacked = factor_s.unstack()

    def _get_top_quantile_weights(cross_section: pd.Series) -> pd.Series:
        """计算单期截面上，头部组合的等权重。"""
        cs_clean = cross_section.dropna()
        if len(cs_clean) < quantiles:
            return pd.Series(0.0, index=cross_section.index)
            
        # 使用 rank(method='first') 防止遇到大量重复值时 qcut 报错
        ranks = cs_clean.rank(method='first')
        try:
            q_bins = pd.qcut(ranks, q=quantiles, labels=False)
            top_q = q_bins.max()
            
            # 选出最高分位数的一组，计算等权权重
            long_stocks = cs_clean[q_bins == top_q]
            weight = 1.0 / len(long_stocks) if len(long_stocks) > 0 else 0.0
            
            weights = pd.Series(weight, index=long_stocks.index)
            # 重新对齐到全市场并填补 0
            return weights.reindex(cross_section.index).fillna(0.0)
        except ValueError:
            return pd.Series(0.0, index=cross_section.index)

    # 计算逐期的持仓权重矩阵
    weights_df = factor_unstacked.apply(_get_top_quantile_weights, axis=1)
    
    # 真实换手率：前后两期绝对权重变化之和的一半（单边换手率）
    # fillna(0) 处理上市/退市/调仓带来的空值
    turnover = weights_df.fillna(0.0).diff().abs().sum(axis=1) / 2.0
    
    turnover.name = "turnover"
    return turnover.dropna()


def calc_t_stat(ic_series: pd.Series) -> tuple[float, float]:
    """IC 序列的 t 统计量及 p-value。"""
    ic_clean = ic_series.dropna()
    if len(ic_clean) < 2:
        return (np.nan, np.nan)
    t_stat, p_value = stats.ttest_1samp(ic_clean, 0)
    return (t_stat, p_value)


def calc_forward_returns(
    market_data: pd.DataFrame,
    periods: list[int] | None = None,
    price_col: str = "close",
) -> dict[int, pd.Series]:
    """根据行情数据计算各期前向收益。
    
    注意：使用 close 计算，隐含假设是 T 日收盘后计算因子，以 T 日收盘价进行交易（理论环境）。
    若需靠近实盘（T+1开盘买入），建议传入的 price_col 对应的数据口径调整为次日开盘价。
    """
    if periods is None:
        periods = [1, 5, 10, 20]

    price = market_data[price_col].unstack()
    result = {}
    for p in periods:
        fwd_ret = price.shift(-(1 + p)) / price.shift(-1) - 1
        result[p] = fwd_ret.stack().rename(f"fwd_ret_{p}")
    return result
