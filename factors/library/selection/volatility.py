"""波动率因子。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.schema import Col, FundamentalCol
from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class Volatility20(BaseFactor):
    name = "volatility_20"
    description = "20日收益率标准差"
    category = "risk"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        close = market_data[Col.CLOSE].unstack(Col.SYMBOL)
        ret = close.pct_change()
        vol = ret.rolling(20).std()
        vol.index.name = Col.DATE
        vol.columns.name = Col.SYMBOL
        return vol


@register_factor
class Volatility5(BaseFactor):
    name = "volatility_5"
    description = "5日收益率标准差"
    category = "risk"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        close = market_data[Col.CLOSE].unstack(Col.SYMBOL)
        ret = close.pct_change()
        vol = ret.rolling(5).std()
        vol.index.name = Col.DATE
        vol.columns.name = Col.SYMBOL
        return vol


@register_factor
class HighLowSpread20(BaseFactor):
    name = "highlow_spread_20"
    description = "20日最高最低价振幅均值"
    category = "risk"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        high = market_data[Col.HIGH].unstack(Col.SYMBOL)
        low = market_data[Col.LOW].unstack(Col.SYMBOL)
        close = market_data[Col.CLOSE].unstack(Col.SYMBOL)
        spread = (high - low) / close
        avg_spread = spread.rolling(20).mean()
        avg_spread.index.name = Col.DATE
        avg_spread.columns.name = Col.SYMBOL
        return avg_spread


@register_factor
class Vff3(BaseFactor):
    name = "vff3"
    description = "Fama-French三因子残差年化波动率"
    category = "risk"

    window = 20

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if fundamental_data is None or fundamental_data.empty:
            raise ValueError("vff3 因子需要 fundamental_data 提供市值和 PB 数据")

        close = market_data[Col.CLOSE].unstack(Col.SYMBOL)
        stock_returns = close.pct_change(fill_method=None)

        size = self._align_to_returns(self._get_size(market_data, fundamental_data), stock_returns)
        bm = self._align_to_returns(self._get_book_to_market(fundamental_data), stock_returns)
        if size.dropna(how="all").empty:
            raise ValueError("vff3 因子市值数据全为空，无法构造 market_return 和 SMB")
        if bm.dropna(how="all").empty:
            raise ValueError("vff3 因子账面市值比数据全为空，无法构造 HML")

        factor_returns = pd.DataFrame(
            {
                "market": self._market_return(stock_returns, size),
                "smb": self._long_short_return(stock_returns, size, ascending=True),
                "hml": self._long_short_return(stock_returns, bm, ascending=False),
            },
            index=stock_returns.index,
        )

        signals = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns, dtype=float)
        for symbol in stock_returns.columns:
            data = pd.concat(
                [stock_returns[symbol].rename("stock"), factor_returns],
                axis=1,
            )
            signals[symbol] = self._rolling_residual_vol(data)

        signals.index.name = Col.DATE
        signals.columns.name = Col.SYMBOL
        return signals

    def _rolling_residual_vol(self, data: pd.DataFrame) -> pd.Series:
        result = pd.Series(np.nan, index=data.index, dtype=float)
        for end in range(self.window, len(data) + 1):
            window_data = data.iloc[end - self.window:end].dropna()
            if len(window_data) < self.window:
                continue

            y = window_data["stock"].to_numpy()
            x = window_data[["market", "smb", "hml"]].to_numpy()
            x = np.column_stack([np.ones(len(x)), x])
            try:
                coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            except np.linalg.LinAlgError:
                continue

            residuals = y - x @ coef
            result.iloc[end - 1] = float(np.std(residuals, ddof=1) * np.sqrt(252))
        return result

    @staticmethod
    def _get_size(
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
    ) -> pd.DataFrame:
        for source in (fundamental_data, market_data):
            for col in ("market_cap", Col.MKT_CAP, "total_mv", "circ_mv"):
                if col in source.columns:
                    return source[col].unstack(Col.SYMBOL)
        raise ValueError("vff3 因子需要市值列: market_cap / mkt_cap / total_mv / circ_mv")

    @staticmethod
    def _get_book_to_market(fundamental_data: pd.DataFrame) -> pd.DataFrame:
        if FundamentalCol.PB not in fundamental_data.columns:
            raise ValueError("vff3 因子需要 fundamental_data 包含 pb 列")
        pb = fundamental_data[FundamentalCol.PB].unstack(Col.SYMBOL)
        return 1.0 / pb.where(pb > 0)

    @staticmethod
    def _market_return(stock_returns: pd.DataFrame, size: pd.DataFrame) -> pd.Series:
        weights = size.where(size > 0)
        weights = weights.where(stock_returns.notna())
        weights = weights.div(weights.sum(axis=1), axis=0)
        return (stock_returns * weights).sum(axis=1, min_count=1)

    @staticmethod
    def _align_to_returns(data: pd.DataFrame, stock_returns: pd.DataFrame) -> pd.DataFrame:
        aligned = data.copy()
        if isinstance(aligned.index, pd.DatetimeIndex) and isinstance(stock_returns.index, pd.DatetimeIndex):
            target_tz = stock_returns.index.tz
            source_tz = aligned.index.tz
            if target_tz is not None and source_tz is None:
                aligned.index = aligned.index.tz_localize(target_tz)
            elif target_tz is None and source_tz is not None:
                aligned.index = aligned.index.tz_convert(None)
            elif target_tz is not None and source_tz is not None and target_tz != source_tz:
                aligned.index = aligned.index.tz_convert(target_tz)
        return aligned.reindex(index=stock_returns.index, columns=stock_returns.columns).ffill()

    @staticmethod
    def _long_short_return(
        stock_returns: pd.DataFrame,
        signal: pd.DataFrame,
        ascending: bool,
    ) -> pd.Series:
        values = []
        for date, ret_row in stock_returns.iterrows():
            sig_row = signal.loc[date]
            aligned = pd.DataFrame({"ret": ret_row, "signal": sig_row}).dropna()
            if len(aligned) < 2:
                values.append(np.nan)
                continue

            low = aligned["signal"].quantile(0.3)
            high = aligned["signal"].quantile(0.7)
            if ascending:
                long_ret = aligned.loc[aligned["signal"] <= low, "ret"].mean()
                short_ret = aligned.loc[aligned["signal"] >= high, "ret"].mean()
            else:
                long_ret = aligned.loc[aligned["signal"] >= high, "ret"].mean()
                short_ret = aligned.loc[aligned["signal"] <= low, "ret"].mean()
            values.append(long_ret - short_ret)

        return pd.Series(values, index=stock_returns.index)
