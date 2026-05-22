"""端到端因子挖掘流水线。"""

from __future__ import annotations

from typing import Type

import pandas as pd
from loguru import logger

from data.base import DataLoader
from data.schema import Col, FundamentalCol
from evaluation.selection.ic import calc_forward_returns, calc_ic_series, calc_icir
from evaluation.selection.layered import layered_backtest, LayeredResult
from evaluation.selection.report import FactorReport
from evaluation.plot import plot_factor_report
from factors.base import BaseFactor
from factors.registry import FactorRegistry
from models.base import BaseModel


class SelectionPipeline:
    """端到端因子挖掘流水线。

    支持两种模式:
    1. 纯因子评估模式 — 直接计算因子并评估 IC/ICIR/分层回测
    2. ML模型模式 — 用多个因子作为特征训练模型，模型预测值作为综合 alpha

    用法::

        pipeline = FactorPipeline()
        pipeline.set_data_loader(LocalLoader(market_path="data.csv"))
        pipeline.add_factors(["momentum_5", "volatility_20"])
        # 模式1: 纯因子评估
        results = pipeline.run()

        # 模式2: ML模型
        pipeline.set_model(TreeModel(engine="lgbm"))
        results = pipeline.run()
    """

    def __init__(
        self,
        forward_periods: list[int] | None = None,
        ic_method: str = "rank",
        n_groups: int = 5,
    ) -> None:
        self._loader: DataLoader | None = None
        self._factor_names: list[str] = []
        self._factor_instances: list[BaseFactor] = []
        self._model: BaseModel | None = None

        self._market_data: pd.DataFrame | None = None
        self._fundamental_data: pd.DataFrame | None = None

        self._forward_periods = forward_periods or [1, 5, 10, 20]
        self._ic_method = ic_method
        self._n_groups = n_groups
        self._financial_statement_delay_trading_days = 2

    # ------------------------------------------------------------------ #
    #  配置方法（链式调用）
    # ------------------------------------------------------------------ #

    def set_data_loader(self, loader: DataLoader) -> SelectionPipeline:
        self._loader = loader
        return self

    def add_factors(
        self,
        factors: list[str | Type[BaseFactor] | BaseFactor],
    ) -> SelectionPipeline:
        """添加因子。

        Parameters
        ----------
        factors : 因子名称列表、因子类列表、或因子实例列表
        """
        for f in factors:
            if isinstance(f, str):
                self._factor_names.append(f)
            elif isinstance(f, type) and issubclass(f, BaseFactor):
                self._factor_instances.append(f())
            elif isinstance(f, BaseFactor):
                self._factor_instances.append(f)
            else:
                raise TypeError(f"不支持的因子类型: {type(f)}")
        return self

    def set_model(self, model: BaseModel) -> SelectionPipeline:
        self._model = model
        return self

    def load_data(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        load_fundamental: bool = False,
    ) -> SelectionPipeline:
        """手动加载数据（也可由 run() 自动调用）。"""
        if self._loader is None:
            raise RuntimeError("请先调用 set_data_loader()")

        logger.info("加载行情数据...")
        self._market_data = self._loader.load_market_data(symbols, start, end)
        logger.info("行情数据: {} 行", len(self._market_data))

        if load_fundamental:
            try:
                logger.info("加载基本面数据...")
                self._fundamental_data = self._loader.load_fundamental_data(symbols, start, end)
                logger.info("基本面数据: {} 行", len(self._fundamental_data))
            except NotImplementedError:
                logger.warning("数据源不支持基本面数据")
        return self

    # ------------------------------------------------------------------ #
    #  执行
    # ------------------------------------------------------------------ #

    def run(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        load_fundamental: bool = False,
        show_plot: bool = True,
    ) -> dict[str, FactorReport] | dict:
        """执行完整流水线。

        Returns
        -------
        - 纯因子模式: dict[factor_name, FactorReport]
        - ML模型模式: dict with keys "model_report", "cv_result", "feature_importance"
        """
        # 1. 加载数据
        if self._market_data is None:
            self.load_data(symbols, start, end, load_fundamental)

        market = self._market_data
        fundamental = self._delay_financial_statement_columns(self._fundamental_data, market)

        # 2. 收集因子实例
        all_factors: list[BaseFactor] = list(self._factor_instances)
        for name in self._factor_names:
            cls = FactorRegistry.get(name)
            all_factors.append(cls())

        if not all_factors:
            raise RuntimeError("未添加任何因子，请先调用 add_factors()")

        # 3. 计算因子值
        logger.info("计算 {} 个因子...", len(all_factors))
        factor_values: dict[str, pd.Series] = {}
        for f in all_factors:
            logger.info("  计算 {} ...", f.name)
            factor_values[f.name] = f.generate_signals(market, fundamental).stack().rename(f.name)

        factor_df = pd.DataFrame(factor_values)

        # 4. 根据模式执行
        if self._model is None:
            return self._run_factor_evaluation(factor_df, market, show_plot)
        else:
            return self._run_model_pipeline(factor_df, market, show_plot)

    def _run_factor_evaluation(
        self,
        factor_df: pd.DataFrame,
        market_data: pd.DataFrame,
        show_plot: bool,
    ) -> dict[str, FactorReport]:
        """纯因子评估模式。"""
        forward_periods = self._forward_periods
        ic_method = self._ic_method
        n_groups = self._n_groups

        reports: dict[str, FactorReport] = {}

        for col in factor_df.columns:
            logger.info("评估因子: {}", col)
            factor_s = factor_df[col].dropna()

            report = FactorReport(
                factor_values=factor_s,
                market_data=market_data,
                forward_periods=forward_periods,
                ic_method=ic_method,
                n_groups=n_groups,
            )
            report.print()
            reports[col] = report

            if show_plot:
                try:
                    ic_s = report.ic_series(forward_periods[1] if len(forward_periods) > 1 else forward_periods[0])
                    lr = report.layered(forward_periods[1] if len(forward_periods) > 1 else forward_periods[0])
                    decay = report.ic_decay()
                    plot_factor_report(ic_s, lr, decay, factor_name=col)
                except Exception as e:
                    logger.warning("绘图失败: {}", e)

        return reports

    def _delay_financial_statement_columns(
        self,
        fundamental_data: pd.DataFrame | None,
        market_data: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        if fundamental_data is None or fundamental_data.empty or market_data is None or market_data.empty:
            return fundamental_data
        if not isinstance(fundamental_data.index, pd.MultiIndex):
            return fundamental_data

        financial_columns = [
            FundamentalCol.ROE,
            FundamentalCol.ROA,
            FundamentalCol.EPS,
            FundamentalCol.REVENUE_GROWTH,
        ]
        financial_columns = [column for column in financial_columns if column in fundamental_data.columns]
        if not financial_columns:
            return fundamental_data

        financial_values = fundamental_data[financial_columns].dropna(how="all")
        if financial_values.empty:
            return fundamental_data

        trading_dates = self._market_trading_dates(market_data)
        if trading_dates.empty:
            return fundamental_data

        index_frame = financial_values.index.to_frame(index=False)
        source_dates = pd.to_datetime(index_frame[Col.DATE])
        delayed_dates = self._shift_to_delayed_trading_dates(source_dates, trading_dates)
        valid = delayed_dates.notna()
        valid_mask = valid.to_numpy()
        if not valid_mask.any():
            result = fundamental_data.copy()
            result.loc[financial_values.index, financial_columns] = pd.NA
            return result

        result = fundamental_data.copy()
        result.loc[financial_values.index, financial_columns] = pd.NA

        delayed_values = financial_values.iloc[valid_mask].copy()
        delayed_index_frame = index_frame.iloc[valid_mask].copy()
        delayed_index_frame[Col.DATE] = delayed_dates.iloc[valid_mask].to_numpy()
        delayed_values.index = pd.MultiIndex.from_frame(delayed_index_frame[[Col.DATE, Col.SYMBOL]])
        delayed_values = delayed_values.groupby(level=[Col.DATE, Col.SYMBOL]).last()

        combined = result.join(delayed_values, how="outer", rsuffix="_delayed")
        for column in financial_columns:
            delayed_column = f"{column}_delayed"
            if delayed_column not in combined.columns:
                continue
            combined[column] = combined[delayed_column].combine_first(combined[column])
            combined = combined.drop(columns=[delayed_column])

        return combined.sort_index()

    @staticmethod
    def _market_trading_dates(market_data: pd.DataFrame) -> pd.DatetimeIndex:
        if not isinstance(market_data.index, pd.MultiIndex):
            return pd.DatetimeIndex([])
        dates = pd.to_datetime(market_data.index.get_level_values(Col.DATE).unique())
        dates = pd.DatetimeIndex(dates).tz_localize(None).normalize()
        return pd.DatetimeIndex(sorted(dates.unique()))

    def _shift_to_delayed_trading_dates(
        self,
        dates: pd.Series,
        trading_dates: pd.DatetimeIndex,
    ) -> pd.Series:
        normalized = pd.to_datetime(dates).dt.tz_localize(None).dt.normalize()
        positions = trading_dates.searchsorted(normalized, side="right")
        positions = positions + self._financial_statement_delay_trading_days
        valid = positions < len(trading_dates)
        delayed = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
        delayed.loc[valid] = trading_dates.take(positions[valid]).to_numpy()
        return delayed

    def _run_model_pipeline(
        self,
        factor_df: pd.DataFrame,
        market_data: pd.DataFrame,
        show_plot: bool,
    ) -> dict:
        """ML 模型模式。"""
        forward_periods = self._forward_periods
        target_period = forward_periods[1] if len(forward_periods) > 1 else forward_periods[0]
        ic_method = self._ic_method
        n_groups = self._n_groups

        # 构建特征矩阵和目标
        fwd_returns = calc_forward_returns(market_data, [target_period])
        y = fwd_returns[target_period]

        # 对齐
        combined = factor_df.join(y.rename("target"), how="inner").dropna()
        X = combined.drop(columns=["target"])
        y_aligned = combined["target"]

        logger.info("特征矩阵: {} x {}", X.shape[0], X.shape[1])

        # 交叉验证
        logger.info("交叉验证...")
        cv_result = self._model.cross_validate(X, y_aligned)
        logger.info("CV val IC: {}", [f"{v:.4f}" for v in cv_result["val_scores"]])

        # 全量训练
        logger.info("全量训练模型...")
        self._model.fit(X, y_aligned)

        # 预测
        pred = pd.Series(self._model.predict(X), index=X.index, name="model_alpha")

        # 评估
        report = FactorReport(
            factor_values=pred,
            market_data=market_data,
            forward_periods=forward_periods,
            ic_method=ic_method,
            n_groups=n_groups,
        )
        report.print()

        fi = self._model.get_feature_importance()
        if fi is not None:
            logger.info("特征重要性:\n{}", fi.head(20).to_string())

        if show_plot:
            try:
                ic_s = report.ic_series(target_period)
                lr = report.layered(target_period)
                decay = report.ic_decay()
                plot_factor_report(ic_s, lr, decay, factor_name="ML Model Alpha")
            except Exception as e:
                logger.warning("绘图失败: {}", e)

        return {
            "model_report": report,
            "cv_result": cv_result,
            "feature_importance": fi,
            "predictions": pred,
        }
