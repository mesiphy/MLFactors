# MLFactors

MLFactors 当前已经不只是“因子挖掘骨架”，而是一个同时覆盖三类研究路径的本地研究仓库：

- A 股选股因子评估：`SelectionPipeline`
- 择时因子评估：`TimingPipeline`
- 美股最小策略回测与报告：`StrategyPipeline`

现在最稳定、最完整的一条链路是美股本地 SQLite 回测示例：

`USStockLocalLoader -> Momentum20 -> SimpleTopKPortfolioManager -> SimpleBacktestExecutionEngine -> SimpleStrategyAnalyzer`

## 当前仓库重点

- `data/local_loader.py` 同时提供 A 股本地目录加载器 `AStockLocalLoader` 和美股 SQLite 加载器 `USStockLocalLoader`
- `pipeline/strategy_runner.py` 串起信号、组合、撮合、分析四段最小可运行策略回测链路
- `run_us_strategy_backtest.py` 可以直接基于 `/home/setsu/workspace/data/cta_orange.db` 跑出 CSV + `report.html`
- `tests/test_strategy_pipeline.py` 覆盖了当前策略链路的端到端测试

## 环境准备

优先使用项目根目录虚拟环境。

```bash
uv venv
source .venv/bin/activate

uv pip install pandas numpy scipy scikit-learn matplotlib loguru duckdb pyarrow plotly vectorbt pytest
```

确认解释器：

```bash
./.venv/bin/python -c "import sys; print(sys.executable)"
```

本仓库默认直接在项目根目录运行脚本，不要求安装成 Python 包。

## 项目结构

```text
MLFactors/
├── backtest/
│   ├── execution.py              # 撮合执行器与 SimulationResult
│   ├── portfolio.py              # 仓位管理器
│   └── qlib_adapter.py           # qlib 适配层（保留）
├── data/
│   ├── base.py                   # DataLoader 抽象基类
│   ├── local_loader.py           # A 股目录 / 美股 SQLite 加载器
│   └── schema.py                 # 统一字段枚举
├── evaluation/
│   ├── selection/                # IC / 分层回测 / 因子报告
│   ├── timimg/                   # 择时评估报告
│   ├── plot.py                   # 选股因子图表
│   └── strategy_analyzer.py      # vectorbt 策略分析与 HTML 报告
├── factors/
│   ├── base.py                   # BaseFactor / BaseTimingFactor
│   ├── registry.py               # 因子注册中心
│   └── library/
│       ├── selection/            # 选股因子
│       └── timing/               # MA Cross、RSI 等择时因子
├── models/                       # 机器学习模型封装
├── pipeline/
│   ├── selection_runner.py       # 选股因子评估流水线
│   ├── timing_runner.py          # 择时流水线
│   └── strategy_runner.py        # 策略回测流水线
├── outputs/
│   ├── strategy/                 # 当前最小美股策略输出
│   ├── strategy_all_universe/    # 全市场候选池示例输出
│   └── timing/                   # 择时报告输出
├── run_hs300_factor_test.py      # A 股本地因子测试脚本
├── run_us_factor_test.py         # 美股本地因子测试脚本
├── run_timing_factor_test.py     # 择时因子测试脚本
├── run_us_strategy_backtest.py   # 美股策略回测脚本
└── tests/
```

## 快速开始

### 1. 跑通当前最完整的美股策略回测

默认参数会读取本地 SQLite 数据库 `/home/setsu/workspace/data/cta_orange.db`，使用：

- 信号：20 日动量
- 选股：Top 3 等权
- 调仓：每周五 `W-FRI`
- 执行：整数股、现金账户、佣金和滑点各 `0.0005`
- 基准：`SPY`、`QQQ`

命令：

```bash
./.venv/bin/python run_us_strategy_backtest.py \
  --db-path /home/setsu/workspace/data/cta_orange.db \
  --output-dir outputs/strategy
```

主要输出：

- `outputs/strategy/stats.csv`
- `outputs/strategy/equity_curve.csv`
- `outputs/strategy/returns.csv`
- `outputs/strategy/positions.csv`
- `outputs/strategy/trades.csv`
- `outputs/strategy/report/report.html`
- `outputs/strategy/report/summary_stats.csv`
- `outputs/strategy/report/vectorbt_stats.csv`

### 2. 在代码中直接调用策略流水线

```python
from backtest import SimpleBacktestExecutionEngine, SimpleTopKPortfolioManager
from data import USStockLocalLoader
from evaluation import SimpleStrategyAnalyzer
from factors.library.selection.momentum import Momentum20
from pipeline import StrategyPipeline

loader = USStockLocalLoader("/home/setsu/workspace/data/cta_orange.db")
market = loader.load_market_data(
    symbols=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "SPY", "QQQ"],
    start="2020-01-01",
    end="2026-01-01",
)

pipeline = StrategyPipeline(
    alpha_model=Momentum20(),
    portfolio_manager=SimpleTopKPortfolioManager(top_k=3, rebalance_frequency="W-FRI"),
    execution_engine=SimpleBacktestExecutionEngine(
        commission=0.0005,
        slippage=0.0005,
        initial_capital=1_000_000.0,
    ),
    analyzer_cls=SimpleStrategyAnalyzer,
)

result = pipeline.run(market_data=market)
print(result["stats"])
```

### 3. A 股选股因子评估

```python
from data import AStockLocalLoader
from pipeline import SelectionPipeline

pipeline = (
    SelectionPipeline()
    .set_data_loader(AStockLocalLoader(data_root="./cache"))
    .add_factors(["momentum_5", "momentum_20", "volatility_20"])
)

reports = pipeline.run(
    symbols=["600036", "600519", "000001"],
    start="2024-01-01",
    end="2024-12-31",
    show_plot=False,
)

print(reports["momentum_5"].summary())
```

如果本地 A 股目录结构已经准备好，也可以直接运行：

```bash
./.venv/bin/python run_hs300_factor_test.py
```

### 4. 美股选股因子评估

默认读取 `/home/setsu/workspace/data/cta_orange.db`，从 SQLite `bars` 表筛选区间内有足够日线数据的美股，并输出因子汇总与图表：

```bash
./.venv/bin/python run_us_factor_test.py \
  --output-dir outputs/us_factor
```

主要输出：

- `outputs/us_factor/selected_symbols.csv`
- `outputs/us_factor/factor_summary.csv`
- `outputs/us_factor/*_report.png`

### 5. 择时因子测试

`run_timing_factor_test.py` 当前默认用合成数据快速跑通，也可以替换成你自己的 `DataLoader`。

```bash
./.venv/bin/python run_timing_factor_test.py
```

## 数据约定

所有数据加载器统一输出 `MultiIndex(date, symbol)` 的 `DataFrame`，字段名使用 `data.schema.Col` / `FundamentalCol`。

常用行情列：

| 字段 | 说明 |
|---|---|
| `date` | 交易日 |
| `symbol` | 证券代码 |
| `open` / `high` / `low` / `close` | OHLC |
| `adj_close` | 复权收盘价（若数据源可提供） |
| `volume` | 成交量 |
| `amount` | 成交额 |

`USStockLocalLoader` 当前从 SQLite `bars` 表读取日线数据，并在存在 `adjust_factor` 时生成 `adj_close`。

## 自定义扩展

### 自定义选股因子

```python
from data.schema import Col
from factors import BaseFactor, register_factor

@register_factor
class MyAlphaFactor(BaseFactor):
    name = "my_alpha"
    description = "量价背离因子"
    category = "custom"

    def generate_signals(self, market_data, fundamental_data=None):
        close = market_data[Col.CLOSE].unstack(Col.SYMBOL)
        volume = market_data[Col.VOLUME].unstack(Col.SYMBOL)
        signals = close.pct_change(5).rolling(20).corr(volume.pct_change(5))
        signals.index.name = Col.DATE
        signals.columns.name = Col.SYMBOL
        return signals
```

### 查看已注册因子

```python
from factors.registry import FactorRegistry

print(FactorRegistry.list())
print(FactorRegistry.list_detail())
```

### 将选股因子直接接入策略回测

策略回测直接复用 `selection` 目录下的因子实现：

```python
from factors.library.selection.momentum import Momentum5

signals = Momentum5().generate_signals(market_data)
```

这样独立因子测试和 `StrategyPipeline` 的信号生成可以共享同一套 `generate_signals()` 逻辑。

## 测试

当前和新增策略链路一致的测试是：

```bash
./.venv/bin/python -m pytest tests/test_strategy_pipeline.py -q
```

说明：

- 该测试当前通过，已覆盖信号、选股、撮合、分析器和报告导出
- `tests/` 全量测试目前仍有旧用例引用已移除的 `LocalLoader`，在文档与 API 完全同步前不应把 `pytest tests/` 当作绿灯标准

## 注意事项

- 当前执行引擎会把 `T` 日目标权重整体 `shift(1)` 后在 `T+1` 日执行，避免同日信号直接同日成交
- 当前选股动量因子使用 `close` 计算信号；如果改用复权价，需要自行审查是否存在前视偏差
- `Momentum5/10/20`、`SimpleTopKPortfolioManager`、`SimpleBacktestExecutionEngine` 都是“最小可运行实现”，更适合快速验证链路，而不是直接用于生产交易

## License

MIT
