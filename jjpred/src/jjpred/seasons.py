"""Season labels associated with SKUs."""

from __future__ import annotations

from calendar import Month
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
    F = auto()
    S = auto()

    @classmethod
    def map_polars(cls, x: Any) -> str:
        if isinstance(x, str):
            if "F" == x:
                return POSeason.F.name
            elif "S" == x:
                return POSeason.S.name

        raise ValueError(f"Cannot parse {x} as {cls}")


def season_given_month(month: Month) -> Season:
    if month in [
        Month.OCTOBER,
        Month.NOVEMBER,
        Month.DECEMBER,
        Month.JANUARY,
    ]:
        return Season.FW
    elif month in [Month.APRIL, Month.MAY, Month.JUNE, Month.JULY]:
        return Season.SS
    else:
        return Season.AS
