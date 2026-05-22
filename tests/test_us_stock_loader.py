"""USStockLocalLoader tests."""

from __future__ import annotations

import sqlite3

import pandas as pd

from data.local_loader import USStockLocalLoader
from data.schema import Col


def test_us_stock_loader_excludes_non_stock_symbols_from_market_data(tmp_path):
    db_path = tmp_path / "cta_orange.db"
    rows = [
        {
            "symbol": "AAPL",
            "market": "US",
            "frequency": "1d",
            "dt": "2024-01-02T00:00:00+00:00",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
            "turnover": 100500.0,
            "adjust_factor": 1.0,
        },
        {
            "symbol": "XAUUSD",
            "market": "US",
            "frequency": "1d",
            "dt": "2024-01-07T00:00:00+00:00",
            "open": 2000.0,
            "high": 2010.0,
            "low": 1990.0,
            "close": 2005.0,
            "volume": 10.0,
            "turnover": 20050.0,
            "adjust_factor": 1.0,
        },
        {
            "symbol": "SPY",
            "market": "US",
            "frequency": "1d",
            "dt": "2024-01-02T00:00:00+00:00",
            "open": 470.0,
            "high": 471.0,
            "low": 469.0,
            "close": 470.5,
            "volume": 1000.0,
            "turnover": 470500.0,
            "adjust_factor": 1.0,
        },
    ]
    with sqlite3.connect(str(db_path)) as conn:
        pd.DataFrame(rows).to_sql("bars", conn, index=False, if_exists="replace")

    result = USStockLocalLoader(db_path).load_market_data()
    symbols = result.index.get_level_values(Col.SYMBOL).unique().tolist()

    assert symbols == ["AAPL"]


def test_us_stock_loader_loads_etfs_from_dedicated_interface(tmp_path):
    db_path = tmp_path / "cta_orange.db"
    rows = [
        {
            "symbol": "SPY",
            "market": "US",
            "frequency": "1d",
            "dt": "2024-01-02T00:00:00+00:00",
            "open": 470.0,
            "high": 471.0,
            "low": 469.0,
            "close": 470.5,
            "volume": 1000.0,
            "turnover": 470500.0,
            "adjust_factor": 1.0,
        },
        {
            "symbol": "XAUUSD",
            "market": "US",
            "frequency": "1d",
            "dt": "2024-01-07T00:00:00+00:00",
            "open": 2000.0,
            "high": 2010.0,
            "low": 1990.0,
            "close": 2005.0,
            "volume": 10.0,
            "turnover": 20050.0,
            "adjust_factor": 1.0,
        },
    ]
    with sqlite3.connect(str(db_path)) as conn:
        pd.DataFrame(rows).to_sql("bars", conn, index=False, if_exists="replace")

    result = USStockLocalLoader(db_path).load_etf_market_data(["SPY", "XAUUSD"])
    symbols = result.index.get_level_values(Col.SYMBOL).unique().tolist()

    assert symbols == ["SPY"]
