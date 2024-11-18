"""A strategy is a group of categories that share: historical periods, historical
data aggregation criteria, and reference categories."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto, unique
from typing import (
    Any,
    NamedTuple,
    Self,
)

import polars as pl


from jjpred.structlike import StructLike
from jjpred.utils.polars import struct_filter


class CategoryType(Enum):
    PRIMARY = auto()
    DEPENDENT = auto()
    ALL = auto()

    def __hash__(self) -> int:
        return int(self.value)


class CategoryTypeMeta(NamedTuple):
    int_index: int


@dataclass
class CalcReport:
    reported_data: list[tuple[str, Any]] = field(
        default_factory=list, kw_only=True
    )
    filters: set[StructLike] = field(
        default_factory=set, compare=False, hash=False, kw_only=True
    )

    def append(self, **kwargs) -> None:  # type: ignore
        for description, value in kwargs.items():
            if isinstance(value, pl.DataFrame):
                self.reported_data.append(
                    (description, struct_filter(value, self.filters))
                )
            else:
                self.reported_data.append((description, value))

    def __or__(self, other: Any) -> Self:
        if other is None:
            return self
        elif isinstance(other, self.__class__):
            data: dict[str, list[Any]] = dict()
            for name, value in self.reported_data + other.reported_data:
                if name in data.keys():
                    data[name].append(value)
                else:
                    data[name] = [value]
            return self.__class__(
                reported_data=list(data.items()),
                filters=self.filters | other.filters,
            )
        else:
            raise TypeError(f"Cannot union: {self=} | {other=}")

    def __ror__(self, other: Any) -> Self:
        if other is None:
            return self
        elif isinstance(other, self.__class__):
            return other | self
        else:
            raise TypeError(f"Cannot union: {other=} | {self=}")
