"""因子注册表 — 自动发现与管理。"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Type

import pandas as pd
from loguru import logger

from factors.base import BaseFactor, BaseTimingFactor


class FactorRegistry:
    """全局因子注册表。"""

    _registry: dict[str, Type[BaseFactor]] = {}

    @classmethod
    def register(cls, factor_cls: Type[BaseFactor]) -> Type[BaseFactor]:
        name = factor_cls.name
        if not name:
            raise ValueError(f"{factor_cls.__name__} 必须定义非空的 name 属性")
        if name in cls._registry:
            logger.warning("因子 '{}' 已注册，将被覆盖 ({})", name, factor_cls.__name__)
        cls._registry[name] = factor_cls
        logger.debug("已注册因子: {}", name)
        return factor_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseFactor]:
        cls._ensure_loaded()
        if name not in cls._registry:
            raise KeyError(f"因子 '{name}' 未注册。可用因子: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """列出所有已注册的选股类因子名称（不含择时因子）。"""
        cls._ensure_loaded()
        return sorted(
            name for name, fcls in cls._registry.items()
            if not issubclass(fcls, BaseTimingFactor)
        )

    @classmethod
    def list_timing(cls) -> list[str]:
        """列出所有已注册的择时类因子名称。"""
        cls._ensure_loaded()
        return sorted(
            name for name, fcls in cls._registry.items()
            if issubclass(fcls, BaseTimingFactor)
        )

    @classmethod
    def list_detail(cls) -> list[dict]:
        """返回所有选股类因子的详细信息列表（不含择时因子）。"""
        cls._ensure_loaded()
        return [
            {"name": n, "category": c.category, "description": c.description}
            for n, c in sorted(cls._registry.items())
            if not issubclass(c, BaseTimingFactor)
        ]

    @classmethod
    def generate_all(
        cls,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame | None = None,
        factor_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """批量生成因子信号，返回 MultiIndex DataFrame（列名=因子名）。"""
        cls._ensure_loaded()

        names = factor_names or cls.list()
        results: dict[str, pd.Series] = {}
        for name in names:
            factor_cls = cls.get(name)
            factor = factor_cls()
            logger.info("生成因子信号: {}", name)
            results[name] = factor.generate_signals(market_data, fundamental_data).stack().rename(name)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------ #
    #  自动发现
    # ------------------------------------------------------------------ #

    _loaded = False

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._loaded:
            return
        cls._discover_builtin()
        cls._loaded = True

    @classmethod
    def _discover_builtin(cls) -> None:
        """自动导入 factors.library 下的所有模块。\n\n        若模块已在 sys.modules 中（例如测试 reset 后重新发现），\n        使用 reload 确保模块级 @register_factor 装饰器重新执行。
        """
        import sys
        import factors.library as lib_pkg

        for importer, full_name, ispkg in pkgutil.walk_packages(
            lib_pkg.__path__, prefix="factors.library."
        ):
            try:
                if full_name in sys.modules:
                    importlib.reload(sys.modules[full_name])
                else:
                    importlib.import_module(full_name)
                logger.debug("已加载内置因子模块: {}", full_name)
            except Exception as e:
                logger.warning("加载因子模块 {} 失败: {}", full_name, e)

    @classmethod
    def discover_extra(cls, paths: list[str | Path]) -> None:
        """显式从额外路径加载因子模块。"""
        for path_str in paths:
            path = Path(path_str)
            if not path.is_dir():
                logger.warning("额外因子路径不存在: {}", path)
                continue
            for py_file in path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                module_name = f"_mlfactors_extra_.{py_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(mod)
                        logger.debug("已加载外部因子: {}", py_file)
                    except Exception as e:
                        logger.warning("加载外部因子 {} 失败: {}", py_file, e)

    @classmethod
    def reset(cls) -> None:
        """重置注册表（主要用于测试）。"""
        cls._registry.clear()
        cls._loaded = False


def register_factor(cls: Type[BaseFactor]) -> Type[BaseFactor]:
    """装饰器 — 将因子类注册到全局 FactorRegistry。

    用法::

        @register_factor
        class MyFactor(BaseFactor):
            name = "my_factor"
            def generate_signals(self, market_data, fundamental_data=None):
                ...
    """
    return FactorRegistry.register(cls)
