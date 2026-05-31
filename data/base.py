"""DataLoader 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from data.schema import Col, FundamentalCol


class DataLoader(ABC):
    """所有数据加载器的基类。

    子类需要实现 ``load_market_data``，并可实现 ``load_data`` 一次性返回
    多张数据表。返回的 DataFrame 必须使用 ``Col`` / ``FundamentalCol`` 中
    定义的列名，并以 ``(date, symbol)`` 为 MultiIndex 或包含 date、symbol 列。
    """

    # ------------------------------------------------------------------ #
    #  抽象接口
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load_market_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """加载行情数据。

        Parameters
        ----------
        symbols : 股票代码列表，None 表示全部
        start, end : 起止日期 (含两端)

        Returns
        -------
        DataFrame，至少包含 Col.market_required() 列
        """
        ...

    def load_fundamental_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """加载基本面数据（可选实现）。"""
        raise NotImplementedError("该数据源不支持基本面数据加载")

    def load_data(
        self,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """一次性加载数据表，返回以数据库表名为 key 的字典。"""
        return {"market": self.load_market_data(symbols, start, end)}

    # ------------------------------------------------------------------ #
    #  通用工具方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _standardize(df: pd.DataFrame, column_mapping: dict[str, str] | None = None) -> pd.DataFrame:
        """列名映射 + 日期解析 + 标准化 symbol 列为字符串。"""
        if column_mapping:
            df = df.rename(columns=column_mapping)

        if Col.DATE in df.columns:
            df[Col.DATE] = pd.to_datetime(df[Col.DATE])

        # 确保 symbol 列始终为字符串类型（CSV 读取时数字代码会被解析为 int）
        if Col.SYMBOL in df.columns:
            df[Col.SYMBOL] = df[Col.SYMBOL].astype(str)

        return df

    @staticmethod
    def _set_index(df: pd.DataFrame) -> pd.DataFrame:
        """设置 (date, symbol) MultiIndex。"""
        if df.index.names == [Col.DATE, Col.SYMBOL]:
            return df
        if {Col.DATE, Col.SYMBOL}.issubset(df.columns):
            return df.set_index([Col.DATE, Col.SYMBOL]).sort_index()
        return df

    @staticmethod
    def _filter(
        df: pd.DataFrame,
        symbols: list[str] | None,
        start: str | date | None,
        end: str | date | None,
    ) -> pd.DataFrame:
        """按 symbols 和日期范围过滤。"""
        if symbols is not None:
            if Col.SYMBOL in df.columns:
                df = df[df[Col.SYMBOL].isin(symbols)]
            elif Col.SYMBOL in df.index.names:
                df = df.loc[df.index.get_level_values(Col.SYMBOL).isin(symbols)]

        if start is not None or end is not None:
            date_col = Col.DATE
            if date_col in df.columns:
                if start is not None:
                    df = df[df[date_col] >= pd.Timestamp(start)]
                if end is not None:
                    df = df[df[date_col] <= pd.Timestamp(end)]
            elif date_col in df.index.names:
                idx = df.index.get_level_values(date_col)
                mask = pd.Series(True, index=df.index)
                if start is not None:
                    mask &= idx >= pd.Timestamp(start)
                if end is not None:
                    mask &= idx <= pd.Timestamp(end)
                df = df[mask]

        return df
