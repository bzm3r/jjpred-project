"""Season labels associated with SKUs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import auto
from typing import Any, Literal, Self, TypedDict
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

    def po_priority(self) -> int:
        match self:
            case POSeason.FW:
                return 1
            case POSeason.SS:
                return 0
            case x:
                raise ValueError(f"No PO priority assigned to season {x}")

    def flip_season(self) -> POSeason:
        match self:
            case POSeason.FW:
                return POSeason.SS
            case POSeason.SS:
                return POSeason.FW
            case _:
                raise ValueError(f"No logic for handling {self}")


class CurrentSeasonDict(TypedDict):
    year: int
    po_season: str


@dataclass
class CurrentSeason:
    year: int
    po_season: POSeason

    def unit_offset_season(self, offset: Literal[-1, 0, 1]) -> CurrentSeason:
        assert len(POSeason) == 2

        year_offset = 0
        match (self.po_season, offset):
            case (_, 0):
                return self
            case (POSeason.SS, -1) | (POSeason.FW, 1):
                year_offset = offset
            case (POSeason.SS, 1) | (POSeason.FW, -1):
                year_offset = 0
            case unhandled_case:
                raise ValueError(f"No logic for handling {unhandled_case}")

        return CurrentSeason(
            self.year + year_offset, self.po_season.flip_season()
        )

    def offset_season(self, offset: int) -> CurrentSeason:
        if offset > 0:
            u = 1
        elif offset < 0:
            u = -1
        elif offset == 0:
            return self
        else:
            raise ValueError(f"No logic for handling {offset=}")

        return self.unit_offset_season(u).offset_season(offset - u)

    def as_dict(self) -> CurrentSeasonDict:
        return {"year": self.year, "po_season": self.po_season.name}
