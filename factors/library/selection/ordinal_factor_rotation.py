"""基于市场变量的有效因子序数回归轮动。"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from data.schema import Col
from factors.base import BaseFactor
from factors.registry import register_factor
from . import alpha158_kline, alpha158_price, alpha158_rolling

try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
except ModuleNotFoundError:
    OrderedModel = None


def _alpha158_factor_class_map() -> dict[str, type[BaseFactor]]:
    return {
        "alpha158_std60": getattr(alpha158_rolling, "Alpha158Std60"),
        "alpha158_klen": alpha158_kline.Alpha158Klen,
        "alpha158_std20": getattr(alpha158_rolling, "Alpha158Std20"),
        "alpha158_std30": getattr(alpha158_rolling, "Alpha158Std30"),
        "alpha158_kup": alpha158_kline.Alpha158Kup,
        "alpha158_high0": alpha158_price.Alpha158High0,
        "alpha158_std10": getattr(alpha158_rolling, "Alpha158Std10"),
        "alpha158_std5": getattr(alpha158_rolling, "Alpha158Std5"),
        "alpha158_max5": getattr(alpha158_rolling, "Alpha158Max5"),
        "alpha158_max10": getattr(alpha158_rolling, "Alpha158Max10"),
    }


@register_factor
class OrdinalFactorRotationEnhance(BaseFactor):
    name = "ordinal_factor_rotation_enhance"
    description = "基于 Alpha158 月频子因子的序数回归轮动"
    category = "composite"

    FACTOR_NAMES = [
        "alpha158_std60",
        "alpha158_klen",
        "alpha158_std20",
        "alpha158_std30",
        "alpha158_kup",
        "alpha158_high0",
        "alpha158_std10",
        "alpha158_std5",
        "alpha158_max5",
        "alpha158_max10",
    ]
    GROWTH_ETFS = ["XLK", "XLY", "XLC"]
    DEFENSIVE_ETFS = ["XLP", "XLU", "XLV"]
    CYCLICAL_ETFS = ["XLF", "XLI", "XLE", "XLB"]

    TRAIN_WINDOW = 60
    WEIGHT_UPDATE_INTERVAL = 1
    TOP_QUANTILE = 0.2
    TOP_FRAC = 0.5
    MIN_AVAILABLE_FACTORS = 5
    MIN_STOCK_FACTORS = 3

    def _compute_factor_returns(
        self,
        factor_dict: dict[str, pd.DataFrame],
        stock_returns_forward: pd.DataFrame,
        top_quantile: float = 0.2,
    ) -> pd.DataFrame:
        result = pd.DataFrame(index=stock_returns_forward.index, columns=list(factor_dict), dtype=float)
        for name, factor in factor_dict.items():
            aligned_factor = factor.reindex_like(stock_returns_forward)
            values = []
            for date, factor_row in aligned_factor.iterrows():
                returns_row = stock_returns_forward.loc[date]
                data = pd.DataFrame({"factor": factor_row, "return": returns_row}).dropna()
                if len(data) < 5:
                    values.append(np.nan)
                    continue
                count = max(1, int(np.floor(len(data) * top_quantile)))
                ordered = data.sort_values("factor")
                short_ret = ordered.head(count)["return"].mean()
                long_ret = ordered.tail(count)["return"].mean()
                values.append(long_ret - short_ret)
            result[name] = values
        return result

    @staticmethod
    def _factor_returns_to_ranks(factor_returns: pd.DataFrame) -> pd.DataFrame:
        return factor_returns.rank(axis=1, ascending=False, method="min")

    def _compute_dynamic_factor_weights(
        self,
        factor_returns: pd.DataFrame,
        factor_ranks: pd.DataFrame,
        market_features: pd.DataFrame,
        train_window: int | None = None,
        update_interval: int | None = None,
    ) -> pd.DataFrame:
        factor_returns = factor_returns.reindex(columns=self.FACTOR_NAMES)
        factor_ranks = factor_ranks.reindex(columns=self.FACTOR_NAMES)
        market_features = market_features.reindex(factor_returns.index)
        weights = pd.DataFrame(0.0, index=factor_returns.index, columns=self.FACTOR_NAMES)
        previous: pd.Series | None = None
        self._ordered_model_update_attempts = 0
        self._ordered_model_update_successes = 0
        train_window = self.TRAIN_WINDOW if train_window is None else train_window
        update_interval = self.WEIGHT_UPDATE_INTERVAL if update_interval is None else update_interval

        for position, date in enumerate(factor_returns.index):
            start = max(0, position - train_window)
            hist_returns = factor_returns.iloc[start:position]
            hist_ranks = factor_ranks.iloc[start:position]
            hist_features = market_features.iloc[start:position]
            current_features = market_features.loc[date]

            should_update = previous is None or position % update_interval == 0
            if not should_update:
                current = previous
            elif len(hist_returns.dropna(how="all")) < max(20, self.MIN_AVAILABLE_FACTORS):
                current = self._fallback_weights(hist_returns, previous)
            elif OrderedModel is None:
                current = self._fallback_weights(hist_returns, previous)
            else:
                self._ordered_model_update_attempts += 1
                current = self._ordered_model_weights(hist_returns, hist_ranks, hist_features, current_features)
                if current is not None:
                    self._ordered_model_update_successes += 1
                if current is None:
                    current = self._fallback_weights(hist_returns, previous)

            weights.loc[date] = current
            previous = current

        return weights

    def _combine_factor_signals(
        self,
        factor_dict: dict[str, pd.DataFrame],
        factor_weights: pd.DataFrame,
    ) -> pd.DataFrame:
        zscores = {
            name: self._cross_sectional_zscore(factor.reindex(factor_weights.index))
            for name, factor in factor_dict.items()
        }
        columns = next(iter(factor_dict.values())).columns
        signals = pd.DataFrame(np.nan, index=factor_weights.index, columns=columns)

        for date in factor_weights.index:
            weight_row = factor_weights.loc[date].reindex(self.FACTOR_NAMES).fillna(0.0)
            numerator = pd.Series(0.0, index=columns)
            denominator = pd.Series(0.0, index=columns)
            counts = pd.Series(0, index=columns)

            for name, weight in weight_row.items():
                if weight <= 0 or name not in zscores:
                    continue
                values = zscores[name].loc[date].reindex(columns)
                mask = values.notna()
                numerator.loc[mask] += values.loc[mask] * weight
                denominator.loc[mask] += weight
                counts.loc[mask] += 1

            valid = (denominator > 0) & (counts >= self.MIN_STOCK_FACTORS)
            signals.loc[date, valid] = numerator.loc[valid] / denominator.loc[valid]

        signals.index.name = Col.DATE
        signals.columns.name = Col.SYMBOL
        return signals

    @staticmethod
    def _unstack_optional(data: pd.DataFrame, column: str) -> pd.DataFrame | None:
        if column not in data.columns:
            return None
        return data[column].unstack(Col.SYMBOL).sort_index()

    @staticmethod
    def _group_relative_return(
        close_all: pd.DataFrame,
        left_symbols: list[str],
        right_symbols: list[str],
        window: int,
    ) -> pd.Series:
        left = [symbol for symbol in left_symbols if symbol in close_all.columns]
        right = [symbol for symbol in right_symbols if symbol in close_all.columns]
        if not left or not right:
            return pd.Series(np.nan, index=close_all.index)
        left_ret = close_all[left].pct_change(window, fill_method=None).mean(axis=1)
        right_ret = close_all[right].pct_change(window, fill_method=None).mean(axis=1)
        return left_ret - right_ret

    @staticmethod
    def _resolve_signal_dates(data: dict[str, pd.DataFrame], fallback_index: pd.Index) -> pd.DatetimeIndex:
        signal_dates = data.get("signal_dates")
        if signal_dates is None:
            return pd.DatetimeIndex(pd.to_datetime(fallback_index)).sort_values()
        if isinstance(signal_dates, pd.DataFrame):
            values = signal_dates.index if Col.DATE not in signal_dates.columns else signal_dates[Col.DATE]
        elif isinstance(signal_dates, pd.Series):
            values = signal_dates
        else:
            values = signal_dates
        return pd.DatetimeIndex(pd.to_datetime(values)).tz_localize(None).unique().sort_values()

    def _build_signal_market_features(
        self,
        close_all: pd.DataFrame,
        volume_all: pd.DataFrame | None,
        etf_symbols: list[object],
    ) -> pd.DataFrame:
        usable_etfs = [symbol for symbol in etf_symbols if symbol in close_all.columns]
        proxy = close_all[usable_etfs] if usable_etfs else close_all
        returns = proxy.pct_change(fill_method=None)
        market_return = returns.mean(axis=1)
        market_level = (1.0 + market_return.fillna(0.0)).cumprod()

        features = pd.DataFrame(index=close_all.index)
        for window in (1, 3, 6):
            features[f"market_ret_{window}m"] = market_level.pct_change(window, fill_method=None)
        for window in (3, 6, 12):
            features[f"market_vol_{window}m"] = (
                market_return.rolling(window, min_periods=max(2, window // 2)).std() * np.sqrt(12)
            )

        rolling_peak = market_level.rolling(12, min_periods=3).max()
        features["market_drawdown_12m"] = market_level / rolling_peak - 1.0

        for window in (1, 3):
            sector_ret = proxy.pct_change(window, fill_method=None)
            features[f"sector_dispersion_{window}m"] = sector_ret.std(axis=1)
            valid_count = sector_ret.notna().sum(axis=1)
            positive_count = sector_ret.gt(0).sum(axis=1)
            features[f"sector_breadth_{window}m"] = positive_count / valid_count.replace(0, np.nan)

        for window in (1, 3):
            features[f"growth_defensive_ret_{window}m"] = self._group_relative_return(
                close_all,
                self.GROWTH_ETFS,
                self.DEFENSIVE_ETFS,
                window,
            )
            features[f"cyclical_defensive_ret_{window}m"] = self._group_relative_return(
                close_all,
                self.CYCLICAL_ETFS,
                self.DEFENSIVE_ETFS,
                window,
            )

        if volume_all is not None:
            volume_proxy = volume_all[[symbol for symbol in usable_etfs if symbol in volume_all.columns]]
            if volume_proxy.empty:
                volume_proxy = volume_all
            total_volume = volume_proxy.sum(axis=1, min_count=1)
            vol_mean = total_volume.rolling(3, min_periods=2).mean()
            vol_std = total_volume.rolling(3, min_periods=2).std()
            features["market_volume_zscore_3m"] = (total_volume - vol_mean) / vol_std.replace(0, np.nan)

        return features.replace([np.inf, -np.inf], np.nan)

    def _ordered_model_weights(
        self,
        hist_returns: pd.DataFrame,
        hist_ranks: pd.DataFrame,
        hist_features: pd.DataFrame,
        current_features: pd.Series,
    ) -> pd.Series | None:
        feature_cols = hist_features.dropna(axis=1, how="all").columns
        if len(feature_cols) == 0:
            return None

        scores = pd.Series(np.nan, index=self.FACTOR_NAMES, dtype=float)
        rank_count = len(self.FACTOR_NAMES)
        cutoff = max(1, int(np.floor(rank_count * self.TOP_FRAC)))

        for factor_name in self.FACTOR_NAMES:
            y = hist_ranks[factor_name]
            data = hist_features[feature_cols].join(y.rename("rank")).replace([np.inf, -np.inf], np.nan)
            data = data.dropna(subset=["rank"])
            data = data[data["rank"].between(1, rank_count)]
            if len(data) < max(30, len(feature_cols) + 5) or data["rank"].nunique() < 2:
                continue

            x = data[feature_cols]
            med = x.median()
            x = x.fillna(med).dropna(axis=1, how="any")
            if x.empty:
                continue
            x_mean = x.mean()
            x_std = x.std(ddof=0).replace(0, np.nan)
            x = ((x - x_mean) / x_std).dropna(axis=1, how="any")
            if x.empty:
                continue

            y = data.loc[x.index, "rank"].astype(int)
            unique_labels = np.sort(y.unique())
            current = current_features.reindex(x.columns).fillna(med.reindex(x.columns))
            current = ((current - x_mean.reindex(x.columns)) / x_std.reindex(x.columns)).fillna(0.0)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = OrderedModel(y, x, distr="logit")
                    fitted = model.fit(method="bfgs", disp=False, maxiter=100)
                    pred = fitted.model.predict(fitted.params, exog=pd.DataFrame([current], columns=x.columns))
            except Exception:
                continue

            probs = np.asarray(pred).reshape(-1)
            model_labels = getattr(fitted.model, "labels", None)
            labels = np.asarray(model_labels, dtype=int) if model_labels is not None else unique_labels
            if len(labels) != len(probs):
                continue
            scores[factor_name] = float(probs[labels <= cutoff].sum())

        scores = scores.clip(lower=0).fillna(0.0)
        if scores.sum() <= 0:
            return None
        return scores / scores.sum()

    def _fallback_weights(
        self,
        hist_returns: pd.DataFrame,
        previous: pd.Series | None,
    ) -> pd.Series:
        mean_returns = hist_returns.reindex(columns=self.FACTOR_NAMES).mean(skipna=True)
        available = mean_returns.dropna()
        if len(available) >= self.MIN_AVAILABLE_FACTORS:
            ranks = available.rank(ascending=False, method="first")
            scores = len(available) - ranks + 1.0
            weights = pd.Series(0.0, index=self.FACTOR_NAMES)
            weights.loc[scores.index] = scores / scores.sum()
            return weights
        if previous is not None and previous.sum() > 0:
            return previous
        return self._equal_weights(self.FACTOR_NAMES)

    @staticmethod
    def _equal_weights(factor_names: list[str]) -> pd.Series:
        weights = pd.Series(1.0 / len(factor_names), index=factor_names, dtype=float)
        return weights

    @staticmethod
    def _cross_sectional_zscore(factor: pd.DataFrame) -> pd.DataFrame:
        lower = factor.quantile(0.01, axis=1)
        upper = factor.quantile(0.99, axis=1)
        clipped = factor.clip(lower=lower, upper=upper, axis=0)
        mean = clipped.mean(axis=1)
        std = clipped.std(axis=1).replace(0, np.nan)
        return clipped.sub(mean, axis=0).div(std, axis=0)

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        market_data = data["market"]
        etf_data = data.get("etf", pd.DataFrame())
        macro_data = data.get("macro", pd.DataFrame())

        close_daily = market_data[Col.CLOSE].unstack(Col.SYMBOL).sort_index()
        signal_dates = self._resolve_signal_dates(data, close_daily.index)
        close = close_daily.reindex(signal_dates)
        stock_symbols = list(close.columns)
        etf_close = pd.DataFrame(index=close.index)
        if etf_data is not None and not etf_data.empty and Col.CLOSE in etf_data.columns:
            etf_close = etf_data[Col.CLOSE].unstack(Col.SYMBOL).sort_index().reindex(signal_dates)
            etf_close = etf_close.loc[:, ~etf_close.columns.isin(stock_symbols)]

        close_all = pd.concat([close, etf_close], axis=1).sort_index()
        volume_daily = self._unstack_optional(market_data, Col.VOLUME)
        volume = volume_daily.reindex(signal_dates) if volume_daily is not None else None
        etf_volume_daily = self._unstack_optional(etf_data, Col.VOLUME) if etf_data is not None and not etf_data.empty else None
        etf_volume = etf_volume_daily.reindex(signal_dates) if etf_volume_daily is not None else None
        if volume is None:
            volume_all = etf_volume
        elif etf_volume is None:
            volume_all = volume
        else:
            etf_volume = etf_volume.loc[:, ~etf_volume.columns.isin(volume.columns)]
            volume_all = pd.concat([volume, etf_volume], axis=1).sort_index()

        if not stock_symbols:
            raise ValueError("ordinal_factor_rotation_enhance 无可用股票标的")
        etf_symbols = list(etf_close.columns)
        if volume_all is not None:
            volume_all = volume_all.reindex(index=close_all.index, columns=close_all.columns)

        factor_dict = self._compute_alpha158_sub_factors(market_data, close)
        available = [name for name, factor in factor_dict.items() if not factor.dropna(how="all").empty]
        if len(available) < self.MIN_AVAILABLE_FACTORS:
            raise ValueError(
                f"ordinal_factor_rotation_enhance 可用子因子不足: {len(available)} < {self.MIN_AVAILABLE_FACTORS}"
            )

        stock_returns_forward = close.pct_change(fill_method=None).shift(-1)
        market_features = self._build_signal_market_features(close_all, volume_all, etf_symbols)
        factor_returns = self._compute_factor_returns(
            factor_dict,
            stock_returns_forward,
            top_quantile=self.TOP_QUANTILE,
        )
        factor_ranks = self._factor_returns_to_ranks(factor_returns)
        factor_weights = self._compute_dynamic_factor_weights(
            factor_returns,
            factor_ranks,
            market_features,
        )
        signals = self._combine_factor_signals(factor_dict, factor_weights)
        signals = signals.reindex(index=close.index, columns=stock_symbols)
        signals.index.name = Col.DATE
        signals.columns.name = Col.SYMBOL

        self.last_signal_dates = signal_dates
        self.last_market_features = market_features
        self.last_factor_dict = factor_dict
        self.last_factor_returns = factor_returns
        self.last_factor_ranks = factor_ranks
        self.last_factor_weights = factor_weights
        self.last_macro_data = macro_data
        return signals

    def _compute_alpha158_sub_factors(
        self,
        market_data: pd.DataFrame,
        close: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        blank = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        factor_classes = _alpha158_factor_class_map()
        factors: dict[str, pd.DataFrame] = {}
        for factor_name in self.FACTOR_NAMES:
            factor_cls = factor_classes.get(factor_name)
            if factor_cls is None:
                factors[factor_name] = blank.copy()
                continue

            factor = factor_cls().generate_signals({"market": market_data})
            if not isinstance(factor, pd.DataFrame) or factor.empty:
                factors[factor_name] = blank.copy()
                continue
            factor = factor.apply(pd.to_numeric, errors="coerce")
            factors[factor_name] = factor.reindex(index=close.index, columns=close.columns)
        return factors
