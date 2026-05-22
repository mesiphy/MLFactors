"""
A 股因子测试脚本（本地数据）
=============================
基于本地落盘数据，对 A 股测试以下三个因子：
  - momentum_5   （5日动量）
  - momentum_20  （20日动量）
  - volatility_20（20日波动率）

用法：
  ./.venv/bin/python run_hs300_factor_test.py

常用参数：
  --data-root    本地数据根目录（与 fetcher 使用的 data_root 一致）
  --max-stocks   最多使用多少只股票，0 表示不限制
  --start / --end 回测区间
  --output-dir   报告输出目录
  --no-plots     只输出汇总 CSV，不保存图表
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")  # 必须在导入 pyplot 前设置，避免无头环境弹窗

from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import duckdb
import matplotlib.pyplot as plt
from loguru import logger

from data.local_loader import AStockLocalLoader
from evaluation.plot import plot_factor_report
from pipeline.selection_runner import SelectionPipeline

DEFAULT_DATA_ROOT = "/home/setsu/workspace/MLFactors/cache"
DEFAULT_START_DATE = "20240101"
DEFAULT_END_DATE = "20260101"
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_FACTORS = ["momentum_5", "momentum_20", "volatility_20"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 A 股选股因子级别测试")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=DEFAULT_END_DATE)
    parser.add_argument("--max-stocks", type=int, default=300, help="0 表示不限制")
    parser.add_argument("--factors", nargs="+", default=DEFAULT_FACTORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot-period", type=int, default=5)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


# ── 1. 从元数据库读取股票列表 ─────────────────────────────────────────────

def get_symbols(
    data_root: str,
    start: str,
    end: str,
    max_stocks: int | None = None,
) -> list[str]:
    """从 meta_data.duckdb 读取回测区间内有效的 A 股代码。

    过滤条件与 AStockLocalLoader 保持一致：
      - 截止日前已上市（list_date <= end）
      - 未在开始日前退市（delist_date IS NULL OR delist_date >= start）
    """
    meta_db = Path(data_root) / "meta_data.duckdb"
    if not meta_db.exists():
        logger.error(
            "meta_data.duckdb 不存在: {}，请先运行数据下载流程", meta_db
        )
        return []

    query = """
        SELECT symbol FROM securities
        WHERE market = 'A_stock'
          AND asset_type = 'stock'
          AND (list_date IS NULL OR list_date <= ?)
          AND (delist_date IS NULL OR delist_date >= ?)
        ORDER BY symbol
    """
    if max_stocks is not None:
        query += f" LIMIT {int(max_stocks)}"

    def _to_dash(d: str) -> str:
        s = str(d).replace("-", "")
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"

    with duckdb.connect(str(meta_db), read_only=True) as conn:
        rows = conn.execute(query, [_to_dash(end), _to_dash(start)]).fetchall()

    symbols = [r[0] for r in rows]
    logger.info("共加载 {} 只有效 A 股（区间 {} ~ {}）", len(symbols), start, end)
    return symbols


# ── 2. 主流程 ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    max_stocks = None if args.max_stocks == 0 else args.max_stocks
    output_dir = args.output_dir if not args.no_plots else None

    # 读取股票列表
    symbols = get_symbols(args.data_root, args.start, args.end, max_stocks=max_stocks)
    if not symbols:
        logger.error("股票列表为空，退出")
        return

    # 构建流水线：AStockLocalLoader 从本地文件加载行情
    pipeline = (
        SelectionPipeline()
        .set_data_loader(AStockLocalLoader(data_root=args.data_root))
        .add_factors(args.factors)
    )

    # run() 内部调用 loader.load_market_data(symbols, start, end)
    # 返回 dict[factor_name, FactorReport]
    reports = pipeline.run(
        symbols=symbols,
        start=args.start,
        end=args.end,
        show_plot=False,  # 手动保存，不弹窗
    )

    # ── 3. 打印汇总 + 保存图表 ───────────────────────────────────────────
    for name, report in reports.items():
        report.print()

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            cfg_periods = report.forward_periods
            if args.plot_period in cfg_periods:
                plot_period = args.plot_period
            else:
                plot_period = cfg_periods[1] if len(cfg_periods) > 1 else cfg_periods[0]

            ic_s  = report.ic_series(plot_period)
            lr    = report.layered(plot_period)
            decay = report.ic_decay()

            fig = plot_factor_report(ic_s, lr, decay, factor_name=name, period=plot_period)
            save_path = output_dir / f"{name}_report.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("图表已保存: {}", save_path)

    # ── 4. 汇总 CSV ──────────────────────────────────────────────────────
    if output_dir is not None:
        import pandas as pd
        rows = []
        for name, report in reports.items():
            df = report.summary()
            df.insert(0, "factor", name)
            rows.append(df)
        summary = pd.concat(rows)
        csv_path = output_dir / "factor_summary.csv"
        summary.to_csv(csv_path)
        logger.info("汇总表已保存: {}", csv_path)
        print("\n" + "=" * 55)
        print("  综合汇总（所有因子 × 所有周期）")
        print("=" * 55)
        print(summary.to_string())


if __name__ == "__main__":
    main()
