"""data 层单元测试 — LocalLoader, schema, DataLoader 工具方法。"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.schema import Col, FundamentalCol
from data.base import DataLoader
from data.local_loader import LocalLoader


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def make_market_df(n_dates: int = 5, n_symbols: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    symbols = [f"S{i:03d}" for i in range(n_symbols)]
    records = []
    rng = np.random.default_rng(0)
    for d in dates:
        for s in symbols:
            records.append({
                Col.DATE: d,
                Col.SYMBOL: s,
                Col.OPEN: 10.0,
                Col.HIGH: 11.0,
                Col.LOW: 9.0,
                Col.CLOSE: float(rng.uniform(9, 11)),
                Col.VOLUME: float(rng.integers(1000, 5000)),
            })
    return pd.DataFrame(records)


# ── Schema 测试 ───────────────────────────────────────────────────────────────

class TestSchema:
    def test_col_market_required(self):
        required = Col.market_required()
        assert Col.DATE in required
        assert Col.SYMBOL in required
        assert Col.CLOSE in required

    def test_fundamental_required(self):
        required = FundamentalCol.required()
        assert FundamentalCol.DATE in required
        assert FundamentalCol.SYMBOL in required


# ── DataLoader 工具方法测试 ───────────────────────────────────────────────────

class TestDataLoaderUtils:
    def _df(self):
        return make_market_df()

    def test_standardize_renames_columns(self):
        df = self._df().rename(columns={Col.CLOSE: "收盘"})
        result = DataLoader._standardize(df, {"收盘": Col.CLOSE})
        assert Col.CLOSE in result.columns

    def test_standardize_parses_date(self):
        df = self._df()
        df[Col.DATE] = df[Col.DATE].astype(str)
        result = DataLoader._standardize(df)
        assert pd.api.types.is_datetime64_any_dtype(result[Col.DATE])

    def test_set_index_creates_multiindex(self):
        df = self._df()
        result = DataLoader._set_index(df)
        assert result.index.names == [Col.DATE, Col.SYMBOL]

    def test_filter_by_symbols(self):
        df = self._df()
        result = DataLoader._filter(df, symbols=["S000"], start=None, end=None)
        assert set(result[Col.SYMBOL].unique()) == {"S000"}

    def test_filter_by_date_range(self):
        df = self._df()
        start = "2024-01-02"
        end = "2024-01-03"
        result = DataLoader._filter(df, symbols=None, start=start, end=end)
        assert result[Col.DATE].min() >= pd.Timestamp(start)
        assert result[Col.DATE].max() <= pd.Timestamp(end)


# ── LocalLoader 测试 ──────────────────────────────────────────────────────────

class TestLocalLoaderCSV:
    def test_load_csv(self, tmp_path):
        df = make_market_df()
        csv_path = tmp_path / "market.csv"
        df.to_csv(csv_path, index=False)

        loader = LocalLoader(market_path=csv_path)
        result = loader.load_market_data()

        assert result.index.names == [Col.DATE, Col.SYMBOL]
        assert Col.CLOSE in result.columns
        assert len(result) == len(df)

    def test_filter_symbols(self, tmp_path):
        df = make_market_df()
        csv_path = tmp_path / "market.csv"
        df.to_csv(csv_path, index=False)

        loader = LocalLoader(market_path=csv_path)
        result = loader.load_market_data(symbols=["S000"])
        symbols = result.index.get_level_values(Col.SYMBOL).unique().tolist()
        assert symbols == ["S000"]

    def test_filter_date_range(self, tmp_path):
        df = make_market_df(n_dates=10)
        csv_path = tmp_path / "market.csv"
        df.to_csv(csv_path, index=False)

        loader = LocalLoader(market_path=csv_path)
        result = loader.load_market_data(start="2024-01-03", end="2024-01-05")
        dates = result.index.get_level_values(Col.DATE)
        assert dates.min() >= pd.Timestamp("2024-01-03")
        assert dates.max() <= pd.Timestamp("2024-01-05")

    def test_column_mapping(self, tmp_path):
        df = make_market_df().rename(columns={Col.CLOSE: "收盘Price"})
        csv_path = tmp_path / "market.csv"
        df.to_csv(csv_path, index=False)

        loader = LocalLoader(market_path=csv_path, column_mapping={"收盘Price": Col.CLOSE})
        result = loader.load_market_data()
        assert Col.CLOSE in result.columns


class TestLocalLoaderParquet:
    def test_load_parquet(self, tmp_path):
        df = make_market_df()
        pq_path = tmp_path / "market.parquet"
        df.to_parquet(pq_path, index=False)

        loader = LocalLoader(market_path=pq_path)
        result = loader.load_market_data()
        assert len(result) == len(df)


class TestLocalLoaderSQLite:
    def test_load_sqlite(self, tmp_path):
        df = make_market_df()
        db_path = tmp_path / "market.db"
        conn = sqlite3.connect(str(db_path))
        df.to_sql("market", conn, index=False, if_exists="replace")
        conn.close()

        loader = LocalLoader(market_path=db_path, market_table="market")
        result = loader.load_market_data()
        assert result.index.names == [Col.DATE, Col.SYMBOL]
        assert len(result) == len(df)

    def test_no_path_raises(self):
        loader = LocalLoader()
        with pytest.raises(ValueError):
            loader.load_market_data()

