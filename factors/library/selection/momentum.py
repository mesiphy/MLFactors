"""动量因子。"""

from __future__ import annotations

import pandas as pd

from data.schema import Col
from factors.base import BaseFactor
from factors.registry import register_factor


@register_factor
class Momentum5(BaseFactor):
    name = "momentum_5"
    description = "5日动量（过去5日涨跌幅）"
    category = "momentum"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        prices = market_data[Col.CLOSE].unstack(Col.SYMBOL)
        signals = prices.pct_change(5)
        signals.index.name = Col.DATE
        signals.columns.name = Col.SYMBOL
        return signals


@register_factor
class Momentum10(BaseFactor):
    name = "momentum_10"
    description = "10日动量"
    category = "momentum"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        prices = market_data[Col.CLOSE].unstack(Col.SYMBOL)
        signals = prices.pct_change(10)
        signals.index.name = Col.DATE
        signals.columns.name = Col.SYMBOL
        return signals


@register_factor
class Momentum20(BaseFactor):
    name = "momentum_20"
    description = "20日动量"
    category = "momentum"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        prices = market_data[Col.CLOSE].unstack(Col.SYMBOL)
        signals = prices.pct_change(20)
        signals.index.name = Col.DATE
        signals.columns.name = Col.SYMBOL
        return signals


@register_factor
class TimeSeriesMomentum252(BaseFactor):
    name = "time_series_momentum_252"
    description = "252日时间序列动量（近12个月收益率）"
    category = "momentum"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        prices = market_data[Col.CLOSE].unstack(Col.SYMBOL)
        signals = prices.pct_change(252, fill_method=None)
        signals.index.name = Col.DATE
        signals.columns.name = Col.SYMBOL
        return signals
