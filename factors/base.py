"""因子抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from data.schema import Col


class BaseFactor(ABC):
    """选股因子基类。

    子类必须定义 ``name`` 属性并实现 ``generate_signals`` 方法。
    因子输出统一为 ``pd.DataFrame``，索引为 ``date``，列为 ``symbol``。

    Attributes
    ----------
    name : 因子唯一标识名
    description : 因子描述
    category : 因子类别（如 "momentum", "value", "quality" 等）
    """

    name: str = ""
    description: str = ""
    category: str = "custom"

    @abstractmethod
    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """生成选股信号矩阵。

        Parameters
        ----------
        data : 以数据库表名为 key 的数据字典，例如
            ``market`` / ``etf`` / ``fundamental`` / ``statement`` / ``macro``。

        Returns
        -------
        pd.DataFrame，索引为 date，列为 symbol，值为因子信号
        """
        ...

    def __repr__(self) -> str:
        return f"<Factor: {self.name} ({self.category})>"


class BaseTimingFactor(BaseFactor):
    """择时因子基类（单股票时序信号）。

    与选股因子 ``BaseFactor`` 的区别：
    - 面向单只股票的时序信号，而非多股票截面排序
    - 实现 ``compute_timing(market_data, symbol)``，返回按日期索引的信号序列
    - 信号约定：正值 = 多头方向，负值 = 空头方向，0 = 空仓
    - ``generate_signals()`` 被禁用，调用将抛出 ``NotImplementedError``

    Attributes
    ----------
    factor_type : 固定为 ``"timing"``，用于从注册表中区分因子类型
    """

    factor_type: str = "timing"
    category: str = "timing"

    @abstractmethod
    def compute_timing(
        self,
        market_data: pd.DataFrame,
        symbol: str,
    ) -> pd.Series:
        """计算择时信号序列。

        Parameters
        ----------
        market_data : 行情数据 DataFrame，MultiIndex(date, symbol)
        symbol : 目标股票代码

        Returns
        -------
        pd.Series，索引为 date，值为信号（正=多头方向，负=空头方向，0=空仓）
        """
        ...

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """择时因子不支持截面调用，请使用 ``compute_timing()``。"""
        raise NotImplementedError(
            f"择时因子 '{self.name}' 不支持截面调用，请使用 compute_timing(market_data, symbol)"
        )

    @staticmethod
    def _get_symbol_close(
        market_data: pd.DataFrame,
        symbol: str,
        price_col: str = Col.CLOSE,
    ) -> pd.Series:
        """从行情数据中提取单股收盘价序列。

        自动处理 symbol 索引层的数据类型不一致问题（例如 CSV 加载后 symbol 变为 int64）。

        Parameters
        ----------
        market_data : MultiIndex(date, symbol) DataFrame 或 date 单索引 DataFrame
        symbol : 股票代码（字符串）
        price_col : 价格列名（默认 ``"close"``）

        Returns
        -------
        pd.Series，索引为 DatetimeIndex，已排序
        """
        if isinstance(market_data.index, pd.MultiIndex):
            level_vals = market_data.index.get_level_values(Col.SYMBOL)
            key: str | int = symbol
            if not (level_vals == symbol).any():
                try:
                    key = level_vals.dtype.type(symbol)
                except (ValueError, TypeError):
                    pass
            close = market_data.xs(key, level=Col.SYMBOL)[price_col]
        else:
            close = market_data[price_col]
        close.index = pd.to_datetime(close.index)
        return close.sort_index()
