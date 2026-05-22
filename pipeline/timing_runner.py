"""择时因子测试流水线。

与选股类 ``FactorPipeline`` 对应，面向单只或多只目标股票的时序信号评估。
支持链式调用配置，最终返回 ``dict[factor_name, dict[symbol, TimingReport]]``。

用法示例
--------
::

    from data.akshare_loader import AKShareLoader
    from pipeline.timing_runner import TimingPipeline

    reports = (
        TimingPipeline()
        .set_data_loader(AKShareLoader())
        .set_symbol(["600036", "000001"])
        .set_benchmark("000300")          # 可选：基准指数
        .add_factors(["ma_cross_timing", "rsi_timing"])
        .run(start="20230101", end="20260101")
    )

    for factor_name, symbol_reports in reports.items():
        for symbol, report in symbol_reports.items():
            report.print()
"""

from __future__ import annotations

from typing import Type

import pandas as pd
from loguru import logger

from data.base import DataLoader
from evaluation.timimg.report import TimingReport
from factors.base import BaseTimingFactor
from factors.registry import FactorRegistry


class TimingPipeline:
    """端到端择时因子测试流水线。

    支持对一只或多只股票批量评估多个择时因子，
    每个（因子, 股票）组合生成一份 ``TimingReport``。

    Parameters
    ----------
    （通过链式调用配置，见各 set_* / add_factors 方法）
    """

    def __init__(
        self,
        price_col: str = "close",
        trading_days: int = 252,
        risk_free: float = 0.0,
    ) -> None:
        self._loader: DataLoader | None = None
        self._symbols: list[str] = []
        self._benchmark_symbol: str | None = None
        self._factor_names: list[str] = []
        self._factor_instances: list[BaseTimingFactor] = []

        self._market_data: pd.DataFrame | None = None

        self._price_col = price_col
        self._trading_days = trading_days
        self._risk_free = risk_free

    # ------------------------------------------------------------------ #
    #  配置方法（链式调用）
    # ------------------------------------------------------------------ #

    def set_data_loader(self, loader: DataLoader) -> TimingPipeline:
        """设置数据加载器（复用现有 DataLoader 体系）。"""
        self._loader = loader
        return self

    def set_symbol(self, symbol: str | list[str]) -> TimingPipeline:
        """设置目标股票代码（支持单个字符串或列表）。"""
        if isinstance(symbol, str):
            self._symbols = [symbol]
        else:
            self._symbols = list(symbol)
        return self

    def set_benchmark(self, symbol: str) -> TimingPipeline:
        """设置基准股票/指数代码（可选）。

        设置后会额外加载该基准的行情数据，用于计算超额收益指标。
        """
        self._benchmark_symbol = symbol
        return self

    def add_factors(
        self,
        factors: list[str | Type[BaseTimingFactor] | BaseTimingFactor],
    ) -> TimingPipeline:
        """添加择时因子。

        Parameters
        ----------
        factors : 以下三种方式均可混合使用：

            - ``str`` — 因子注册名（须已通过 ``@register_factor`` 注册且为择时因子）
            - 类 — ``BaseTimingFactor`` 子类
            - 实例 — ``BaseTimingFactor`` 实例（可携带自定义参数）
        """
        for f in factors:
            if isinstance(f, str):
                self._factor_names.append(f)
            elif isinstance(f, type) and issubclass(f, BaseTimingFactor):
                self._factor_instances.append(f())
            elif isinstance(f, BaseTimingFactor):
                self._factor_instances.append(f)
            else:
                raise TypeError(
                    f"不支持的因子类型: {type(f)}，请传入 BaseTimingFactor 子类或实例"
                )
        return self

    # ------------------------------------------------------------------ #
    #  数据加载
    # ------------------------------------------------------------------ #

    def load_data(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> TimingPipeline:
        """预加载行情数据（也可由 ``run()`` 自动调用）。

        Parameters
        ----------
        start, end : 行情数据起止日期（含两端），格式 ``"YYYYMMDD"`` 或 ``"YYYY-MM-DD"``
        """
        if self._loader is None:
            raise RuntimeError("请先调用 set_data_loader()")
        if not self._symbols:
            raise RuntimeError("请先调用 set_symbol()")

        # 需要加载的全部代码（目标股票 + 可选基准）
        symbols_to_load = list(self._symbols)
        if self._benchmark_symbol and self._benchmark_symbol not in symbols_to_load:
            symbols_to_load.append(self._benchmark_symbol)

        logger.info("加载行情数据: {} 个代码 ...", len(symbols_to_load))
        self._market_data = self._loader.load_market_data(symbols_to_load, start, end)
        logger.info("行情数据加载完成: {} 行", len(self._market_data))
        return self

    # ------------------------------------------------------------------ #
    #  执行
    # ------------------------------------------------------------------ #

    def run(
        self,
        start: str | None = None,
        end: str | None = None,
        show_plot: bool = False,
    ) -> dict[str, dict[str, TimingReport]]:
        """执行完整择时因子测试流水线。

        Parameters
        ----------
        start, end : 如果尚未调用 load_data()，会先自动加载数据
        show_plot : 暂未实现，预留接口

        Returns
        -------
        嵌套字典：``dict[factor_name, dict[symbol, TimingReport]]``

        Raises
        ------
        RuntimeError : 未设置 data_loader / symbol / 因子 时抛出
        TypeError    : add_factors() 传入非择时因子时抛出
        """
        # 1. 确保数据已加载
        if self._market_data is None:
            self.load_data(start, end)

        if not self._symbols:
            raise RuntimeError("请先调用 set_symbol()")

        # 2. 收集因子实例
        all_factors: list[BaseTimingFactor] = list(self._factor_instances)
        for name in self._factor_names:
            factor_cls = FactorRegistry.get(name)
            if not issubclass(factor_cls, BaseTimingFactor):
                raise TypeError(
                    f"因子 '{name}' 不是择时因子（BaseTimingFactor 子类），"
                    "请使用 FactorPipeline 进行选股因子评估。"
                )
            all_factors.append(factor_cls())

        if not all_factors:
            raise RuntimeError("未添加任何因子，请先调用 add_factors()")

        market = self._market_data
        bm_data = market if self._benchmark_symbol else None

        # 3. 逐因子 × 逐股票评估
        results: dict[str, dict[str, TimingReport]] = {}

        for factor in all_factors:
            logger.info("评估择时因子: {}", factor.name)
            factor_results: dict[str, TimingReport] = {}

            for symbol in self._symbols:
                logger.info("  计算 {} @ {}", factor.name, symbol)
                try:
                    signal = factor.compute_timing(market, symbol)
                except Exception as exc:
                    logger.warning(
                        "因子 '{}' 在 '{}' 上计算失败，已跳过: {}",
                        factor.name, symbol, exc,
                    )
                    continue

                report = TimingReport(
                    factor_values=signal,
                    market_data=market,
                    symbol=symbol,
                    benchmark_data=bm_data,
                    benchmark_symbol=self._benchmark_symbol,
                    price_col=self._price_col,
                    trading_days=self._trading_days,
                    risk_free=self._risk_free,
                )
                report.print()
                factor_results[symbol] = report

            results[factor.name] = factor_results

        return results
