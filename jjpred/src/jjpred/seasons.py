"""Season labels associated with SKUs."""

from __future__ import annotations

from enum import auto
from typing import Any, Self
from jjpred.utils.polars import EnumLike


class Season(EnumLike):
    FW = auto()
    """Fall/Winter"""
    SS = auto()
    """Spring/Summer"""
    AS = auto()
    """All Season"""

    @classmethod
    def map_polars(cls, x: Any) -> str:
        if isinstance(x, str):
            if "F" in x or "FD" in x:
                if "S" in x:
                    return cls.AS.name
                else:
                    return cls.FW.name
            elif "S" in x:
                return cls.SS.name

        raise ValueError(f"Cannot parse {x} as {cls}")

    @classmethod
    def all_seasons(cls) -> list[Self]:
        return [x for x in cls]


class POSeason(EnumLike):
    FW = auto()
    """Fall/Winter"""
    SS = auto()
    """Spring/Summer"""

    @classmethod
    def map_polars(cls, x: Any) -> str:
        if isinstance(x, str):
            if x in ["F", "FW"]:
                return POSeason.FW.name
            elif x in ["S", "SS"]:
                return POSeason.SS.name

        raise ValueError(f"Cannot parse {x} as {cls}")
