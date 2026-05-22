请基于现有因子框架，单独实现一个新的 Python 因子文件。

该因子用于实现“基于市场变量的有效因子序数回归轮动模型”。核心思想来自《宏观变量控制下的有效因子轮动》：不直接预测单个因子的未来收益绝对值，而是预测每个有效因子下一期在所有有效因子中的相对排名概率；排名靠前概率越高，该因子下一期权重越高。原论文使用 13 个有效因子，并采用序数回归法预测因子相对排名，再用“排名前 1/2 的概率”作为赋权依据。

请严格适配如下框架接口：

@register_factor
class OrdinalFactorRotation(BaseFactor):
    name = "ordinal_factor_rotation"
    description = "基于市场变量的有效因子序数回归轮动"
    category = "composite"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        ...

输入只有：
1. market_data:
   - 包含股票和 ETF 的行情数据，不会提前区分股票和 ETF。
   - 至少可能包含 close、open、high、low、volume、amount、turnover、mkt_cap 等字段。
   - index 结构与现有框架一致，symbol 需要通过 unstack(Col.SYMBOL) 展开。
2. fundamental_data:
   - 包含股票基本面数据。
   - 可能包含 roe、roa、eps、revenue、pb、market_cap 等字段。
   - 不一定包含 ETF 基本面数据。
   - 如果某些基本面字段缺失，需要尽量优雅处理，不能让整个因子崩溃，除非必要字段完全不可用。

输出必须是：
- pd.DataFrame
- index 为日期
- columns 为股票代码
- values 为最终综合因子值
- index.name = Col.DATE
- columns.name = Col.SYMBOL

注意：
- 输出的是股票层面的最终综合因子信号，不是因子权重，也不是因子收益。
- 如果框架默认“因子值越大越看多”，则最终输出也应当是越大越看多。
- ETF 只用于构造市场变量，不应出现在最终输出股票池中。
- 股票和 ETF 在 market_data 里混在一起，需要在因子内部区分。

一、整体实现目标

在一个单独因子类中完成以下流程：

1. 从 market_data 中识别股票与 ETF。
2. 使用 ETF 或市场行情构造市场状态变量 market_features。
3. 使用股票行情与基本面数据计算 13 个有效子因子。
4. 根据每个子因子构造历史因子多空收益。
5. 将每期 13 个因子收益转为横截面排名。
6. 使用市场变量作为自变量，用序数回归预测每个子因子下一期排名靠前概率。
7. 将排名靠前概率转为 13 个子因子的动态权重。
8. 在每个日期，将 13 个子因子标准化后按动态权重合成为最终股票信号。
9. 返回最终股票信号矩阵。

二、股票与 ETF 区分逻辑

market_data 中股票和 ETF 不会提前区分，请在因子内部提供可配置识别逻辑。

请实现方法：

_is_etf_symbol(symbol: str) -> bool

默认可以使用以下规则之一或组合：
1. 如果 symbol 在预设 ETF 列表中，则认为是 ETF。
2. 预设 ETF 列表包括美股行业 ETF：
   - XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC
   - 也可包括 SPY, QQQ, IWM, DIA 作为市场代理。
3. 如果 symbol 不在 ETF 列表中，则默认认为是股票。
4. 代码中应允许用户通过类变量配置 ETF_SYMBOLS。

最终：
- ETF symbols 用于构造 market_features。
- stock symbols 用于计算 13 个股票因子和最终输出。

三、需要实现的 13 个有效子因子

请在因子内部实现或尽量实现以下 13 个子因子。

1. ROE
   - 使用 fundamental_data 中的 roe 字段。
   - 正向因子，越大越好。

2. ΔROA
   - 使用 roa 的变化量。
   - 可以用当前 roa 减去上一期 roa。
   - 正向因子，越大越好。

3. ΔROE
   - 使用 roe 的变化量。
   - 可以用当前 roe 减去上一期 roe。
   - 正向因子，越大越好。

4. ΔEPS
   - 使用 eps 的变化量。
   - 可以用当前 eps 减去上一期 eps。
   - 正向因子，越大越好。

5. 主营业务收入增长率
   - 使用 revenue 或 operating_revenue 字段。
   - 可以计算同比或环比增长；若无法同比，则用 pct_change 作为近似。
   - 正向因子，越大越好。

6. PB
   - 使用 pb 字段。
   - 注意 PB 通常是负向估值因子，越低越好。
   - 因此最终正向化时应使用 -PB，或者 direction=-1。

7. 一个月日均换手率
   - 使用 turnover 字段。
   - 若无 turnover，可用 volume / shares 或 volume 的相对指标近似。
   - 原论文中这是情绪类因子，方向请做成可配置。
   - 默认 direction=-1，即低换手更好；但代码中允许修改。

8. 三个月日均换手率
   - 约 63 日 turnover 均值。
   - 默认 direction=-1。

9. 一个月反转
   - 约 21 日收益率的相反数。
   - raw momentum = close.pct_change(21)
   - reversal_1m = -raw momentum
   - 正向因子，越大表示过去跌得越多，反转预期越强。

10. 三个月反转
   - 约 63 日收益率的相反数。
   - 正向因子。

11. 六个月反转
   - 约 126 日收益率的相反数。
   - 正向因子。

12. DIF
   - MACD 中的 DIF。
   - EMA12 - EMA26。
   - 方向默认正向，越大越好。

13. DEA
   - MACD 中的 DEA。
   - DIF 的 EMA9。
   - 方向默认正向，越大越好。

实现要求：
- 每个子因子输出都应是 DataFrame，index 为日期，columns 为股票。
- 所有子因子最终都应正向化，即值越大越看多。
- 对缺失字段要做兼容处理：
  - 如果某个子因子无法计算，则该子因子可以整列为 NaN，并在后续权重计算中剔除或赋权为 0。
  - 不要因为一个子因子缺失导致整个因子失败，除非可用子因子数量低于最低要求。
- 类中设置 MIN_AVAILABLE_FACTORS，例如至少需要 5 个可用子因子。

四、市场变量构造逻辑

请实现方法：

_build_market_features(
    close_all: pd.DataFrame,
    volume_all: pd.DataFrame | None,
    etf_symbols: list[str],
) -> pd.DataFrame

市场变量只使用 ETF 行情构造。如果 ETF 数量不足，也可以退化为使用全部标的或股票等权收益构造市场代理。

至少构造以下变量：

1. 市场整体趋势：
   - market_ret_5d
   - market_ret_21d
   - market_ret_63d

2. 市场波动率：
   - market_vol_21d
   - market_vol_63d

3. 市场回撤：
   - market_drawdown_63d

4. 板块收益离散度：
   - sector_dispersion_21d
   - sector_dispersion_63d

5. 板块上涨宽度：
   - sector_breadth_21d
   - sector_breadth_63d

6. 板块平均相关性：
   - sector_avg_corr_21d
   - sector_avg_corr_63d

7. 成长相对防御：
   - growth_defensive_ret_21d
   - growth_defensive_ret_63d
   - 成长板块默认 XLK、XLY、XLC
   - 防御板块默认 XLP、XLU、XLV

8. 周期相对防御：
   - cyclical_defensive_ret_21d
   - cyclical_defensive_ret_63d
   - 周期板块默认 XLF、XLI、XLE、XLB
   - 防御板块默认 XLP、XLU、XLV

9. 板块轮动强度：
   - top3_minus_bottom3_sector_ret_21d
   - top3_minus_bottom3_sector_ret_63d

10. 成交量情绪：
   - market_volume_zscore_21d
   - 如果 volume 数据可用，则计算 ETF 总成交量的 21 日 z-score。

市场变量要求：
- 使用历史滚动数据计算。
- 不能使用未来数据。
- 输出 index 为日期。
- 如果某些 ETF 不存在，则自动跳过。
- 如果 ETF 太少，相关性、离散度等变量应返回 NaN 或降级处理。

五、因子多空收益计算逻辑

请实现方法：

_compute_factor_returns(
    factor_dict: dict[str, pd.DataFrame],
    stock_returns_forward: pd.DataFrame,
    top_quantile: float = 0.2,
) -> pd.DataFrame

逻辑：

1. 对每个子因子，在每个日期：
   - 取当日因子值。
   - 取下一期股票收益。
   - 删除因子值或收益缺失的股票。
   - 按因子值排序。
   - 做多最高 top 20%。
   - 做空最低 bottom 20%。
   - 因子收益 = 多头平均收益 - 空头平均收益。

2. 因为所有子因子已正向化，所以统一使用“高因子值做多，低因子值做空”。

3. 输出 factor_returns:
   - index 为日期。
   - columns 为子因子名称。
   - values 为该期因子多空收益。

4. 注意时间对齐：
   - t 日因子值只能对应 t+1 或下一调仓期收益。
   - 不能用同一天未来收益。
   - 如果是日频框架，可以默认使用 close-to-close 的下一日收益：
     forward_return = close.pct_change().shift(-1)
   - 如果希望月频轮动，可以在类中提供 REBALANCE_FREQ 参数，例如 "M"，先将因子值和市场变量取月末，再计算下一月收益。

六、因子排名构造逻辑

请实现方法：

_factor_returns_to_ranks(factor_returns: pd.DataFrame) -> pd.DataFrame

逻辑：
- 对每个日期，将所有可用子因子的收益从高到低排序。
- 收益最高排名为 1。
- 收益最低排名为 K。
- K 是当期可用因子数量。
- 如果需要 OrderedModel 类别稳定，建议保留完整 13 个因子的排名；缺失因子跳过或填 NaN。

七、序数回归轮动逻辑

请实现核心方法：

_compute_dynamic_factor_weights(
    factor_returns: pd.DataFrame,
    factor_ranks: pd.DataFrame,
    market_features: pd.DataFrame,
) -> pd.DataFrame

逻辑：

1. 使用滚动窗口训练。
   - train_window 默认 252 个交易日或 60 个调仓期，具体取决于 REBALANCE_FREQ。
   - 如果是月频，默认 60 个月。
   - 如果是日频，默认 252 日或 504 日。
   - 允许类变量配置 TRAIN_WINDOW。

2. 对每个日期 t：
   - 使用 t 之前的历史窗口训练。
   - 使用 t 日可见 market_features 预测 t+1 期子因子排名。
   - 对每个子因子单独训练一个序数回归模型。

3. 单个子因子的序数回归：
   - y 为该子因子过去每期的排名，1 表示最好。
   - X 为对应历史日期的 market_features。
   - 使用 Ordered Logit 或 Ordered Probit。
   - 推荐使用 statsmodels.miscmodels.ordinal_model.OrderedModel。
   - 回归前对 X 做标准化。
   - OrderedModel 不要手动添加常数项。
   - 如果模型拟合失败、样本不足、类别数量不足，则该子因子的 score 设为 NaN 或 0。

4. 预测：
   - 对每个子因子，预测其在 t+1 期的排名概率：
     P(rank=1), P(rank=2), ..., P(rank=K)
   - 计算 score：
     score_i = P(rank <= floor(K * top_frac))
   - top_frac 默认 0.5。
   - 例如 13 个因子时，score_i = P(rank <= 6)。

5. 权重归一化：
   - score 小于 0 的值截断为 0。
   - score 为 NaN 的因子权重设为 0。
   - 如果 score 总和大于 0：
     weight_i = score_i / sum(score)
   - 如果所有模型失败或所有 score 为 0：
     回退为可用子因子等权。

6. 输出 factor_weights:
   - index 为日期。
   - columns 为 13 个子因子名称。
   - values 为该日期用于合成股票信号的子因子权重。

八、最终股票信号合成逻辑

请实现方法：

_combine_factor_signals(
    factor_dict: dict[str, pd.DataFrame],
    factor_weights: pd.DataFrame,
) -> pd.DataFrame

逻辑：

1. 对每个子因子，在每个日期做横截面标准化：
   - winsorize 或 clip 极端值，可选。
   - z-score 标准化：
     z = (x - mean) / std
   - 如果 std 为 0，则该日期该因子信号为 NaN。

2. 对每个日期：
   - 取该日期的 13 个子因子 z-score。
   - 取该日期的动态因子权重。
   - 最终信号 = sum(weight_i * zscore_factor_i)。

3. 若某股票某些子因子缺失：
   - 仅使用该股票有值的子因子。
   - 可以对剩余权重重新归一化。
   - 如果该股票有效子因子数量太少，则输出 NaN。

4. 输出 signals:
   - index 为日期。
   - columns 为股票 symbol。
   - values 为最终综合因子值。
   - 数值越大越看多。

九、框架接口要求

请参考以下风格编写代码：

@register_factor
class HighLowSpread20(BaseFactor):
    name = "highlow_spread_20"
    description = "20日最高最低价振幅均值"
    category = "risk"

    def generate_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        high = market_data[Col.HIGH].unstack(Col.SYMBOL)
        low = market_data[Col.LOW].unstack(Col.SYMBOL)
        close = market_data[Col.CLOSE].unstack(Col.SYMBOL)
        spread = (high - low) / close
        avg_spread = spread.rolling(20).mean()
        avg_spread.index.name = Col.DATE
        avg_spread.columns.name = Col.SYMBOL
        return avg_spread

请按照同样的框架实现：

@register_factor
class OrdinalFactorRotation(BaseFactor):
    name = "ordinal_factor_rotation"
    description = "基于市场变量的有效因子序数回归轮动"
    category = "composite"

    ETF_SYMBOLS = [...]
    FACTOR_NAMES = [...]
    TRAIN_WINDOW = ...
    TOP_QUANTILE = 0.2
    TOP_FRAC = 0.5
    REBALANCE_FREQ = None 或 "M"
    MIN_AVAILABLE_FACTORS = 5

    def generate_signals(...):
        ...

十、异常处理与降级逻辑

1. 如果 fundamental_data 为 None：
   - 仍然计算纯行情类子因子：
     一个月日均换手率、三个月日均换手率、一个月反转、三个月反转、六个月反转、DIF、DEA。
   - 基本面类因子置为 NaN。
   - 如果可用因子少于 MIN_AVAILABLE_FACTORS，则 raise ValueError。

2. 如果 ETF 数量不足：
   - 使用所有标的或股票等权收益构造简化 market_features。
   - 板块相关变量可以为 NaN。
   - 后续训练时自动删除全 NaN 特征。

3. 如果 statsmodels 不可用：
   - 可以提供 fallback：
     使用过去窗口内各子因子平均收益排名作为权重；
     或者使用因子等权。
   - 但主实现应优先使用 OrderedModel。

4. 如果某个日期模型无法训练：
   - 回退为上一期权重。
   - 如果没有上一期权重，则回退为可用因子等权。

5. 所有滚动计算必须避免未来函数。

十一、代码质量要求

1. 使用 pandas / numpy 实现。
2. 使用 statsmodels 的 OrderedModel 实现序数回归。
3. 提供类型注解。
4. 函数拆分清晰。
5. 每个关键步骤添加中文注释。
6. 不要写成 notebook，直接写成可放入因子库的单个 Python 文件。
7. 不要依赖外部数据下载。
8. 不要假设具体股票代码格式，除了 ETF_SYMBOLS 可配置。
9. 保证返回结果与现有框架格式一致。

十二、最终目标

生成一个独立因子文件，实现：

market_data + fundamental_data
    -> 区分股票和 ETF
    -> ETF 构造市场变量
    -> 股票构造 13 个有效子因子
    -> 计算每个子因子的历史多空收益
    -> 将子因子收益转成排名
    -> 用市场变量训练序数回归模型
    -> 预测各子因子下一期排名靠前概率
    -> 生成动态子因子权重
    -> 合成股票层面的最终综合因子信号
    -> 返回 DataFrame[date, symbol]

请优先保证逻辑正确、时间对齐严格、接口兼容现有 BaseFactor 框架。