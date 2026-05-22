"""
美股因子测试脚本（本地 SQLite 数据）
====================================
基于本地 ``cta_orange.db``，对美股测试 vff3 因子。

用法：
  ./.venv/bin/python run_us_factor_test.py

常用参数：
  --db-path      SQLite 数据库路径
  --start / --end 回测区间
  --max-stocks   最多使用多少只股票，0 表示不限制
  --output-dir   报告输出目录
  --plot-period  只保存指定周期图表；不指定则保存全部评估周期
  --no-plots     只输出汇总 CSV，不保存图表
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from scipy import stats

from data.local_loader import USStockLocalLoader
from data.schema import Col
from evaluation.plot import plot_factor_report
from evaluation.selection.ic import calc_icir, calc_t_stat
from pipeline.selection_runner import SelectionPipeline


DEFAULT_FACTORS = ["vff3"]
DEFAULT_DB_PATH = Path("/home/setsu/workspace/data/cta_orange.db")
DEFAULT_START_DATE = "2016-01-04"
DEFAULT_END_DATE = "2026-04-06"
DEFAULT_OUTPUT_DIR = Path("outputs/us_factor_vff3_all")
DEFAULT_PLOT_PERIOD = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行美股选股因子级别测试")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--market", default="US")
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=DEFAULT_END_DATE)
    parser.add_argument("--max-stocks", type=int, default=0, help="0 表示不限制")
    parser.add_argument("--min-observations", type=int, default=80)
    parser.add_argument("--factors", nargs="+", default=DEFAULT_FACTORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot-period", type=int, default=DEFAULT_PLOT_PERIOD, help="不指定则保存全部评估周期图表")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def get_symbols(
    db_path: Path,
    market: str,
    start: str,
    end: str,
    max_stocks: int | None = None,
    min_observations: int = 80,
) -> list[str]:
    """从 SQLite bars 表读取区间内有足够日线数据的美股代码。"""
    if not db_path.exists():
        logger.error("SQLite 数据库不存在: {}", db_path)
        return []

    conditions = ["market = ?", "frequency = '1d'"]
    params: list[object] = [market]
    if start:
        conditions.append("date(dt) >= ?")
        params.append(str(pd.Timestamp(start).date()))
    if end:
        conditions.append("date(dt) <= ?")
        params.append(str(pd.Timestamp(end).date()))

    query = f"""
        SELECT symbol, COUNT(*) AS n_obs
        FROM bars
        WHERE {" AND ".join(conditions)}
        GROUP BY symbol
        HAVING n_obs >= ?
        ORDER BY symbol
    """
    params.append(int(min_observations))
    if max_stocks is not None and max_stocks > 0:
        query += " LIMIT ?"
        params.append(int(max_stocks))

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(query, params).fetchall()

    symbols = [row[0] for row in rows]
    logger.info(
        "共加载 {} 只美股（market={}, 区间 {} ~ {}, min_observations={}）",
        len(symbols),
        market,
        start,
        end,
        min_observations,
    )
    return symbols


def save_reports(
    reports: dict,
    output_dir: Path,
    save_plots: bool,
    plot_period: int | None,
    benchmark_data: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, report in reports.items():
        summary_df = report.summary()
        summary_df.insert(0, "factor", name)
        rows.append(summary_df)

        if save_plots:
            cfg_periods = list(report.forward_periods)
            if plot_period is not None:
                if plot_period in cfg_periods:
                    plot_periods = [plot_period]
                else:
                    fallback = cfg_periods[1] if len(cfg_periods) > 1 else cfg_periods[0]
                    logger.warning(
                        "plot_period={} 不在评估周期 {} 中，改用 {}",
                        plot_period,
                        cfg_periods,
                        fallback,
                    )
                    plot_periods = [fallback]
            else:
                plot_periods = cfg_periods

            for selected_period in plot_periods:
                ic_s = report.ic_series(selected_period)
                layered = report.layered(selected_period)
                decay = report.ic_decay()
                benchmark_cumulative = benchmark_cumulative_returns(
                    benchmark_data if benchmark_data is not None else report.market_data,
                    index=layered.cumulative_returns.index,
                )

                fig = plot_factor_report(
                    ic_s,
                    layered,
                    decay,
                    benchmark_cumulative=benchmark_cumulative,
                    factor_name=name,
                    period=selected_period,
                )
                save_path = output_dir / f"{name}_{selected_period}d_report.png"
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                if plot_period is not None:
                    legacy_path = output_dir / f"{name}_report.png"
                    fig.savefig(legacy_path, dpi=150, bbox_inches="tight")
                    logger.info("图表已保存: {}", legacy_path)
                plt.close(fig)
                logger.info("图表已保存: {}", save_path)

    summary = pd.concat(rows)
    csv_path = output_dir / "factor_summary.csv"
    summary.to_csv(csv_path)
    logger.info("汇总表已保存: {}", csv_path)

    yearly_ic = build_yearly_ic_summary(reports)
    yearly_ic_path = output_dir / "yearly_ic.csv"
    yearly_ic.to_csv(yearly_ic_path, index=False)
    logger.info("分年度 IC 表已保存: {}", yearly_ic_path)

    print("\n" + "=" * 55)
    print("  美股因子综合汇总（所有因子 x 所有周期）")
    print("=" * 55)
    print(summary.to_string())


def build_yearly_ic_summary(reports: dict) -> pd.DataFrame:
    """按年份汇总完整回测产生的逐日 IC 序列。"""
    records = []
    for factor_name, report in reports.items():
        for period in report.forward_periods:
            ic_series = report.ic_series(period).dropna()
            if ic_series.empty:
                continue

            for year, yearly_ic in ic_series.groupby(ic_series.index.year):
                if len(yearly_ic) < 2:
                    continue

                t_stat, _ = calc_t_stat(yearly_ic)
                t_stat = t_stat / (period ** 0.5)
                df = len(yearly_ic) - 1
                p_value = stats.t.sf(abs(t_stat), df) * 2 if df > 0 else float("nan")

                records.append(
                    {
                        "factor": factor_name,
                        "year": int(year),
                        "period": int(period),
                        "n": int(len(yearly_ic)),
                        "IC_mean": round(float(yearly_ic.mean()), 6),
                        "IC_std": round(float(yearly_ic.std()), 6),
                        "ICIR": round(float(calc_icir(yearly_ic, period, annualize=True)), 6),
                        "t_stat": round(float(t_stat), 6),
                        "p_value": round(float(p_value), 6),
                        "IC_positive_ratio": round(float((yearly_ic > 0).mean()), 6),
                    }
                )

    yearly_ic = pd.DataFrame.from_records(records)
    if not yearly_ic.empty:
        yearly_ic = yearly_ic.sort_values(["factor", "year", "period"]).reset_index(drop=True)
    return yearly_ic


def benchmark_cumulative_returns(
    market_data: pd.DataFrame,
    symbols: tuple[str, ...] = ("SPY", "QQQ"),
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """计算基准累计收益，用于因子报告图。"""
    if market_data.empty or Col.CLOSE not in market_data.columns:
        return pd.DataFrame()

    close = market_data[Col.CLOSE].unstack(Col.SYMBOL)
    present_symbols = [symbol for symbol in symbols if symbol in close.columns]
    if not present_symbols:
        return pd.DataFrame()

    returns = close[present_symbols].pct_change(fill_method=None)
    if index is not None:
        returns = returns.reindex(index)
    cumulative = (1 + returns.fillna(0.0)).cumprod() - 1
    cumulative.index.name = Col.DATE
    return cumulative


def main() -> int:
    args = parse_args()
    max_stocks = None if args.max_stocks == 0 else args.max_stocks

    symbols = get_symbols(
        db_path=args.db_path,
        market=args.market,
        start=args.start,
        end=args.end,
        max_stocks=max_stocks,
        min_observations=args.min_observations,
    )
    if not symbols:
        logger.error("股票列表为空，退出")
        return 1

    selected_symbols_path = args.output_dir / "selected_symbols.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(symbols, name="symbol").to_csv(selected_symbols_path, index=False)
    logger.info("股票列表已保存: {}", selected_symbols_path)

    pipeline = (
        SelectionPipeline()
        .set_data_loader(USStockLocalLoader(db_path=args.db_path, market=args.market))
        .add_factors(args.factors)
    )

    reports = pipeline.run(
        symbols=symbols,
        start=args.start,
        end=args.end,
        load_fundamental=True,
        show_plot=False,
    )

    benchmark_data = USStockLocalLoader(
        db_path=args.db_path,
        market=args.market,
    ).load_etf_market_data(["SPY", "QQQ"], args.start, args.end)

    save_reports(
        reports=reports,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
        plot_period=args.plot_period,
        benchmark_data=benchmark_data,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
