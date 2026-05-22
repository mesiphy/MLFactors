请检查并修改当前 OrdinalFactorRotation 因子实现。当前代码整体框架可以保留，但需要修复若干明显问题，重点是：symbol 类型一致性、基本面因子计算、PB 清洗、换手率代理、序数回归标签稳定性、REBALANCE_FREQ 未使用、市场特征缺失值处理。

请优先在当前文件内修改，不要改动回测框架接口。最终仍然保持：

@register_factor
class OrdinalFactorRotation(BaseFactor):
    name = "ordinal_factor_rotation"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        ...

输出仍然必须是：
- pd.DataFrame
- index 为日期
- columns 为股票 symbol
- values 为最终股票综合因子值
- index.name = Col.DATE
- columns.name = Col.SYMBOL

一、修复 symbol 类型问题

当前代码中存在：

symbols = [str(symbol) for symbol in close_all.columns]
etf_symbols = [symbol for symbol in symbols if self._is_etf_symbol(symbol)]
stock_symbols = [symbol for symbol in symbols if symbol not in etf_symbols]
close = close_all[stock_symbols]

这会把原始 columns 强制转成 str，如果 close_all.columns 原本不是 str 类型，后续索引可能 KeyError。

请改为：

all_symbols = list(close_all.columns)
etf_symbols = [s for s in all_symbols if self._is_etf_symbol(str(s))]
stock_symbols = [s for s in all_symbols if s not in etf_symbols]

要求：
- ETF 判断时可以 str(s).upper()
- DataFrame 列索引必须保留原始 symbol 对象
- 所有 close_all[stock_symbols]、close_all[etf_symbols] 都使用原始 symbol

二、修复 volume_all 对齐问题

当前 volume_all 通过 _unstack_optional 得到后，没有强制对齐 close_all。

请在 generate_signals 中加入：

if volume_all is not None:
    volume_all = volume_all.reindex(index=close_all.index, columns=close_all.columns)

确保 volume_all 与 close_all 日期和列一致。

三、修复基本面数据日期对齐

当前 _fundamental_panel 使用：

panel.reindex(index=close.index, columns=close.columns).ffill()

如果 fundamental_data 的日期不是交易日，reindex 会先丢掉非交易日基本面数据，导致 ffill 无效。

请修改为更稳健的对齐方式：

1. 从 fundamental_data[column].unstack(Col.SYMBOL) 得到 panel
2. panel 按日期排序
3. 先将 panel.index 与 close.index 做 union
4. 在 union index 上 reindex 后 ffill
5. 最后再 reindex 到 close.index 和 close.columns

示例逻辑：

panel = panel.sort_index()
full_index = panel.index.union(close.index)
panel = panel.reindex(full_index).sort_index().ffill()
panel = panel.reindex(index=close.index, columns=close.columns)

同时保留 pd.to_numeric(errors="coerce")。

四、修复 delta_roa / delta_roe / delta_eps 计算

当前代码在基本面数据 ffill 到日频后直接做：

roa.diff()
roe.diff()
eps.diff()

这会导致大多数日期为 0，只有基本面更新日跳变，因子质量很差。

请改为更稳健的近似计算：

delta_roa = roa - roa.shift(63)
delta_roe = roe - roe.shift(63)
delta_eps = eps - eps.shift(63)

说明：
- 63 个交易日近似一个季度
- 如果后续能识别财报频率，可以扩展成真实季度变化
- 当前先使用 63 日差分，避免日频 ffill 后 1 日 diff 失真

五、修复 revenue_growth fallback

当前 _revenue_growth 使用：

revenue.pct_change(fill_method=None)

如果 revenue 已经 ffill 到日频，这同样会导致大多数日期为 0。

请改为：

revenue.pct_change(63, fill_method=None)

或者更稳健地提供类变量：

FUNDAMENTAL_DELTA_WINDOW = 63

并用该变量计算：
- delta_roa
- delta_roe
- delta_eps
- revenue_growth fallback

六、修复 PB 清洗

当前代码：

"pb": -pb if pb is not None else blank.copy()

如果 PB <= 0，会产生错误的极端正向信号。

请改为：

pb_clean = pb.where(pb > 0)
pb_factor = -pb_clean

要求：
- PB <= 0 的位置设为 NaN
- 保持 PB 为负向估值因子，即最终正向化后为 -PB

七、修复换手率代理

当前 _build_turnover_proxy 中，如果存在 Col.AMOUNT，会直接返回成交额 panel。这会把成交额当换手率，强烈偏向大市值股票。

请改为：
- 如果存在真实 turnover 字段，则直接使用
- 如果使用 amount 或 volume 作为代理，则必须相对化：
  proxy = panel / panel.rolling(252, min_periods=20).mean()
- amount 和 volume 都要这样处理
- 避免除以 0

建议逻辑：

if column in (Col.TURNOVER, "turnover"):
    return panel
if column in (Col.AMOUNT, "amount", Col.VOLUME, "volume"):
    base = panel.rolling(252, min_periods=20).mean()
    return panel / base.replace(0, np.nan)

八、修复 sector_breadth 缺失值问题

当前代码：

(sector_ret > 0).mean(axis=1)

如果某行全 NaN，会被当成全 False，返回 0，错误地表示没有板块上涨。

请改为：

valid_count = sector_ret.notna().sum(axis=1)
positive_count = sector_ret.gt(0).sum(axis=1)
breadth = positive_count / valid_count.replace(0, np.nan)

要求：
- 全 NaN 时返回 NaN，而不是 0

九、明确 market_vol 是否年化

当前 market_vol_21d / market_vol_63d 是滚动日收益标准差，没有年化。

请统一改成年化波动率：

rolling_std * np.sqrt(252)

并在注释中说明这是年化波动率。

十、处理 REBALANCE_FREQ 未使用问题

当前类变量 REBALANCE_FREQ 定义了但没有使用，容易误导。

请二选一：

方案 A：暂时不实现月频，则删除 REBALANCE_FREQ，或注释明确说明当前版本只支持日频信号、每 WEIGHT_UPDATE_INTERVAL 日更新权重。

方案 B：实现 REBALANCE_FREQ="M" 的月频路径。

优先选择方案 A，保持修改简单：
- 删除 REBALANCE_FREQ 类变量，或者保留但注释 “目前未启用”
- 不要让用户误以为已经支持月频重采样

十一、提高序数回归标签稳定性

当前 rank_count 使用：

rank_count = int(hist_ranks.max(axis=1).median())

这会导致可用因子数量变化时 cutoff 不稳定。

请改为固定：

rank_count = len(self.FACTOR_NAMES)
cutoff = max(1, int(np.floor(rank_count * self.TOP_FRAC)))

对于 13 个因子，TOP_FRAC=0.5 时，cutoff=6。

十二、处理 OrderedModel 类别映射不稳定问题

当前代码使用：

labels = np.array(getattr(fitted.model, "labels", np.arange(1, len(probs) + 1)), dtype=float)
scores[factor_name] = float(probs[labels <= cutoff].sum())

不同 statsmodels 版本中 labels 不一定可靠，而且如果训练样本缺失某些排名类别，概率类别映射可能不稳定。

请做更稳健处理：

1. 训练前确保 y 是 int 类型，且 rank 在 1 到 len(FACTOR_NAMES) 之间。
2. 如果 y.nunique() < 2，跳过。
3. 预测后，优先使用 fitted.model.labels，如果可用则根据 labels <= cutoff 求和。
4. 如果 labels 不可用，则假设 probs 顺序对应 sorted(y.unique())，而不是 1..len(probs)。
5. 不要简单用 np.arange(1, len(probs)+1)，因为当历史类别不是连续 1..K 时会错。

建议：

unique_labels = np.sort(y.unique())
model_labels = getattr(fitted.model, "labels", None)
if model_labels is not None:
    labels = np.asarray(model_labels, dtype=int)
else:
    labels = unique_labels

if len(labels) != len(probs):
    continue

score = probs[labels <= cutoff].sum()

十三、可选：将序数标签改成“分数越大越好”

当前 rank=1 表示最好，rank=13 表示最差。OrderedModel 可以工作，但方向解释较别扭。

不强制修改。如果修改，请保持 score 逻辑等价：
- rank 越小越好
- 仍然用 P(rank <= cutoff) 作为好因子概率

十四、增强 fallback 权重

当前 fallback 使用历史平均因子收益排名赋权，可以保留。

但请确保：
- 如果 previous 可用，优先使用 previous 还是 mean_returns，请保持当前逻辑即可
- 如果 available 因子少于 MIN_AVAILABLE_FACTORS，则 previous 可用时返回 previous
- 最后才回退等权

十五、保证最终信号不包含 ETF

最终 signals 必须只包含 stock_symbols，不包含 ETF。

generate_signals 最后保留：

signals = signals.reindex(index=close.index, columns=stock_symbols)

但 stock_symbols 必须是原始 symbol 对象，不要强制 str。

十六、添加必要注释

请在关键修改处添加简洁中文注释，说明：
- 为什么不能把 symbol 强制转 str
- 为什么基本面变化用 63 日差分而不是 1 日 diff
- 为什么 PB <= 0 要置 NaN
- 为什么 amount / volume 代理换手率需要相对化
- 为什么 sector_breadth 全 NaN 要返回 NaN

十七、修改后请做最小检查

请完成代码修改后进行以下检查：

1. 静态导入检查：
   - 确认文件可以 import
2. 检查没有明显语法错误
3. 如果项目有测试命令，请运行相关测试
4. 如果无法运行测试，请说明原因
5. 简要列出完成的修复项

十八、不要做的事

1. 不要修改 BaseFactor 接口。
2. 不要改动回测框架。
3. 不要引入外部数据下载。
4. 不要把输出改成权重或因子收益。
5. 不要让 ETF 出现在最终输出 columns 中。
6. 不要删除原有序数回归主逻辑。

## 2026-05-17 修复记录

本次修改文件：`factors/library/selection/ordinal_factor_rotation.py`。

完成的修复项：

- 保留 `close_all.columns` 中的原始 symbol 对象，仅在 ETF 判断时使用 `str(symbol).upper()`，避免非字符串 symbol 触发列索引不匹配。
- `volume_all` 在 `generate_signals()` 中显式对齐到 `close_all` 的日期和列。
- 基本面面板改为先使用基本面日期与交易日 union 后前向填充，再回到交易日索引，避免非交易日基本面数据在 `reindex(close.index)` 时被丢弃。
- `delta_roa`、`delta_roe`、`delta_eps` 改为 `FUNDAMENTAL_DELTA_WINDOW=63` 日差分；收入 fallback 增长率也改为 63 日 `pct_change`。
- PB 因子先将 `PB <= 0` 清洗为 `NaN`，再使用 `-PB` 正向化。
- `amount` / `volume` 换手率代理改为相对自身 252 日滚动均值，避免直接把成交额或成交量当作换手率；真实 `turnover` 字段仍直接使用。
- `sector_breadth_*` 改为 `positive_count / valid_count`，全 NaN 时返回 NaN，不再误记为 0。
- `market_vol_21d`、`market_vol_63d` 改为年化波动率，乘以 `sqrt(252)`。
- 移除误导性的 `REBALANCE_FREQ` 类变量，保留日频信号和 `WEIGHT_UPDATE_INTERVAL=21` 的权重更新方式。
- 序数回归 cutoff 固定使用 `len(FACTOR_NAMES)` 计算，`TOP_FRAC=0.5` 时 cutoff 为 6。
- 序数回归训练标签过滤为 `1..len(FACTOR_NAMES)` 的整数排名，并在预测后优先使用 `fitted.model.labels`；缺失时使用 `sorted(y.unique())` 做概率类别映射，避免假设类别总是连续 `1..K`。
- 保持最终 `signals` 只按原始 `stock_symbols` 输出，ETF 不进入最终 columns。

验证命令：

```bash
./.venv/bin/python -m py_compile factors/library/selection/ordinal_factor_rotation.py
./.venv/bin/python -c "from factors.library.selection.ordinal_factor_rotation import OrdinalFactorRotation; print(OrdinalFactorRotation.name, len(OrdinalFactorRotation.FACTOR_NAMES))"
./.venv/bin/python -m pytest tests/test_factors.py -q
```

额外运行了非字符串 symbol 的最小检查，构造股票 symbol `1`、`2` 和 ETF symbol `SPY`，确认输出列为 `[1, 2]`，ETF 未进入最终信号。
