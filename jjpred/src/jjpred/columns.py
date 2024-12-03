"""Tools for identifying and extracting the right columns from the
``All Marketplace All SKU Categories`` file.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import (
    Any,
    Protocol,
    Self,
    runtime_checkable,
)

import re
import polars as pl
import calendar as cal

from jjpred.channel import Channel
from jjpred.parse.patternmatch import (
    CompiledPattern,
    PatternMatchResult,
    ReMatchResult,
    PatternMatcher,
    ReMatcher,
    ReMatchCondition,
    StringPattern,
)
from jjpred.channel import Platform
from jjpred.structlike import StructLike
from jjpred.utils.typ import (
    Additive,
    ScalarOrList,
    as_list,
)


@runtime_checkable
class ValueParser[T](Protocol):
    def __call__(self, *args: Any, **kwds: Any) -> T:
        raise NotImplementedError()


# type ValueParser[T] = Callable[[Any], T]


def parse_int(x: Any) -> int:
    if x is not None and x != "":
        return int(x)
    else:
        return 0


type Transform = Callable[[dict[str, str], str], str | None]


def use_dict(d: dict[str, str], key: str) -> str:
    return d[key]


def use_dict_and_channel(d: dict[str, str], key: str) -> str:
    if channel := Channel.from_dict(d):
        if key == "country_flag" or key.startswith("country_"):
            return str(channel.country_flag)
        return str(getattr(channel, key))
    return f"No channel from: {d}"


def use_key(d: dict[str, str], key: str) -> str:
    return key


def constant(constant: str, d: dict[str, str], key: str) -> str:
    return constant


class Transformer(ValueParser[str | None]):
    """Convert the parsed representation of a column name into its final column
    name."""

    default: Transform = use_dict

    def __init__(self, default: Transform) -> None:
        super().__init__()
        self.default = default

    def get_default(self) -> Any:
        return self.default

    def __call__(self, d: dict[str, str], key: str) -> str | None:
        return self.default(d, key)


class LabelDefn(Additive):
    """Definition of the label of a column."""

    name: str
    """The final name of the column."""

    matcher: PatternMatcher
    """The matcher used to identify the column."""

    transformers: defaultdict[str, Transform]
    """The transformers used to convert the matched parts of the original column
    label into a final column label."""

    def __init__(
        self,
        name: str,
        matcher: PatternMatcher | type[StructLike],
        default_transform: Transform | None = None,
        transformers: defaultdict[str, Transform] | None = None,
    ):
        self.name = name
        if isinstance(matcher, PatternMatcher):
            self.matcher = matcher
        else:
            self.matcher = matcher.matcher

        if transformers is not None:
            self.transformers = transformers
        else:
            if default_transform is None:
                default_transform = Transformer(partial(constant, name))
            self.transformers = defaultdict(lambda: default_transform, {})

    def __match__(
        self, strings: ScalarOrList[str]
    ) -> PatternMatchResult | None:
        return self.matcher.apply(strings)

    def match_and_transform(
        self, strings: ScalarOrList[str]
    ) -> PatternMatchResult | None:
        if pm := self.__match__(strings):
            result = PatternMatchResult({})
            for key in pm.keys():
                if transformed := self.transformers[key](pm.data, key):
                    result[key] = transformed
                else:
                    result[key] = pm[key]
            return result

    @classmethod
    def combine(cls, left: Self, right: Self) -> Self:
        return cls(
            f"{left.name}-{right.name}",
            left.matcher + right.matcher,
            transformers=left.transformers | right.transformers,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"name: {self.name}, "
            f"matcher: {self.matcher.__repr__()}, "
            f"transformers: {self.transformers.__repr__()}"
            ")"
        )


@dataclass
class Label:
    """A representation of a column label derived from its label definition and
    input strings representing the column."""

    name: str
    match_result: PatternMatchResult

    @classmethod
    def from_defn(
        cls,
        defn: LabelDefn,
        strings: ScalarOrList[str],
        match_parts: list[str],
    ) -> Self | None:
        if match_result := defn.match_and_transform(strings):
            if len(match_parts) > 0:
                name = " ".join([match_result.get(y, "") for y in match_parts])
            else:
                name = defn.name
            return cls(name, match_result)


class ColumnDefn:
    """A column's definition consists of its label's defintion and its
    Polars datatype."""

    name: str
    label_defn: LabelDefn
    dtype: type[pl.DataType]

    def __init__(
        self,
        label_defns: ScalarOrList[LabelDefn],
        dtype: type[pl.DataType],
        match_parts: list[str],
    ):
        label_defns = as_list(label_defns)
        if result := LabelDefn.sum(label_defns):
            self.label_defn = result
        else:
            raise Exception(f"Could not combine {label_defns=}")

        self.name = self.label_defn.name

        self.dtype = dtype
        self.match_parts = match_parts

    def match_raw(
        self, raw_label: ScalarOrList[str], index: int
    ) -> Column | None:
        if label := Label.from_defn(
            self.label_defn, raw_label, self.match_parts
        ):
            return Column(
                label.name,
                [label],
                {self.name: label},
                index,
                self.dtype,
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"name: {self.name}, "
            f"label_defns: {self.label_defn.__repr__()}"
            ")"
        )


@dataclass
class Column:
    """An identified column."""

    name: str
    labels: ScalarOrList[Label]
    label_dict: dict[str, Label]
    index: int
    dtype: type[pl.DataType]
    str_repr: str = field(default_factory=str)

    def __post_init__(self):
        self.labels = as_list(self.labels)
        self.str_repr = " ".join([label.name for label in self.labels])

    def is_compound(self) -> bool:
        return isinstance(self.labels, list)

    def __len__(self) -> int:
        if isinstance(self.labels, list):
            return len(self.labels)
        else:
            return 1

    def __str__(self) -> str:
        return self.str_repr

    def __repr__(self) -> str:
        return self.__str__()


CATEGORY_LABEL_DEFN = LabelDefn(
    "category",
    ReMatcher.from_pattern(
        "category_label_defn",
        CompiledPattern(
            StringPattern("cat_sku").named("category"),
            ReMatchResult,
            flags=re.IGNORECASE,
        ),
        ReMatchCondition.WideAll,
    ),
)

PRINT_LABEL_DEFN = LabelDefn(
    "print",
    ReMatcher.from_pattern(
        "print_label_defn",
        CompiledPattern(
            StringPattern("print_sku").named("print"),
            ReMatchResult,
            flags=re.IGNORECASE,
        ),
        ReMatchCondition.WideAll,
    ),
)


SIZE_LABEL_DEFN = LabelDefn(
    "size",
    ReMatcher.from_pattern(
        "size_label_defn",
        CompiledPattern(
            StringPattern("size").named("size"),
            ReMatchResult,
            flags=re.IGNORECASE,
        ),
        ReMatchCondition.WideAll,
    ),
)

A_SKU_LABEL_DEFN = LabelDefn(
    "a_sku",
    ReMatcher.from_pattern(
        "a_sku_label_defn",
        CompiledPattern(
            StringPattern("adjust_msku").named("sku"),
            ReMatchResult,
            flags=re.IGNORECASE,
        ),
        ReMatchCondition.WideAll,
    ),
)

M_SKU_LABEL_DEFN = LabelDefn(
    "m_sku",
    ReMatcher.from_pattern(
        "m_sku_label_defn",
        CompiledPattern(
            StringPattern("msku").fragmentlike().named("m_sku"),
            ReMatchResult,
            flags=re.IGNORECASE,
        ),
        ReMatchCondition.WideAll,
    ),
)

HISTORY_SHEET_CHANNEL_LABEL_DEFN = LabelDefn(
    "channel",
    ReMatcher.from_pattern(
        "channel_label_defn",
        CompiledPattern(
            StringPattern(r"\bSales Channel$").named("channel"),
            ReMatchResult,
            flags=re.IGNORECASE,
        ),
        ReMatchCondition.WideAll,
    ),
)

IN_STOCK_RATIO_SHEET_CHANNEL_LABEL_DEFN = LabelDefn(
    "channel",
    ReMatcher.from_pattern(
        "channel_label_defn",
        CompiledPattern(
            StringPattern(r"\bWarehouse Location$").named("channel"),
            ReMatchResult,
            flags=re.IGNORECASE,
        ),
        ReMatchCondition.WideAll,
    ),
)

HISTORY_YEAR_MONTH_LABEL_DEFN = LabelDefn(
    "year",
    ReMatcher.from_pattern(
        "year_label_defn",
        CompiledPattern(
            StringPattern(r"\d{4}")
            .named("year")
            .concatenate(
                StringPattern(r"\s*\w*\s*").no_capture().zero_or_more(),
                StringPattern().any_of(*(cal.month_abbr[1:])).named("month"),
            ),
            ReMatchResult,
            flags=re.IGNORECASE,
        ),
        ReMatchCondition.WideAll,
    ),
    transformers=defaultdict(lambda: Transformer(use_dict), {}),
)

IN_STOCK_RATIO_YEAR_MONTH_STOCK_DAYS_LABEL_DEFN = LabelDefn(
    "year_month_stock_days",
    ReMatcher.from_pattern(
        "year_month_stock_ratio_label_defn",
        CompiledPattern(
            StringPattern(r"\d{4}")
            .named("year")
            .concatenate(
                StringPattern(r" In-Stock Days ").no_capture(),
                StringPattern().any_of(*(cal.month_abbr[1:])).named("month"),
            ),
            ReMatchResult,
            flags=re.IGNORECASE,
        ),
        ReMatchCondition.WideAll,
    ),
    transformers=defaultdict(lambda: Transformer(use_dict), {}),
)

INV_CHANNEL_LABEL_DEFN = LabelDefn(
    "channel",
    Platform.amazon_matcher(),
    default_transform=use_dict_and_channel,
)


@dataclass
class ColumnDefns:
    id_cols: list[ColumnDefn]
    data_cols: list[ColumnDefn]


HISTORY_SHEET_COLUMN_DEFNS = ColumnDefns(
    [
        # ComponentDefn(category_defn, pl.String, []),
        # ComponentDefn(print_defn, pl.String, []),
        # ComponentDefn(size_defn, pl.String, []),
        ColumnDefn(A_SKU_LABEL_DEFN, pl.String, []),
        ColumnDefn(HISTORY_SHEET_CHANNEL_LABEL_DEFN, pl.String, []),
    ],
    [
        ColumnDefn(HISTORY_YEAR_MONTH_LABEL_DEFN, pl.Int64, ["year", "month"]),
    ],
)

INVENTORY_SHEET_COLUMN_DEFNS = ColumnDefns(
    [
        # ComponentDefn(category_defn, pl.String, []),
        # ComponentDefn(print_defn, pl.String, []),
        # ComponentDefn(size_defn, pl.String, []),
        ColumnDefn(A_SKU_LABEL_DEFN, pl.String, []),
    ],
    [
        ColumnDefn(
            INV_CHANNEL_LABEL_DEFN, pl.Int64, ["platform", "country_flag"]
        )
    ],
)

IN_STOCK_RATIO_SHEET_COLUMN_DEFNS = ColumnDefns(
    [
        # ComponentDefn(category_defn, pl.String, []),
        # ComponentDefn(print_defn, pl.String, []),
        # ComponentDefn(size_defn, pl.String, []),
        ColumnDefn(M_SKU_LABEL_DEFN, pl.String, []),
        ColumnDefn(A_SKU_LABEL_DEFN, pl.String, []),
        ColumnDefn(IN_STOCK_RATIO_SHEET_CHANNEL_LABEL_DEFN, pl.String, []),
    ],
    [
        ColumnDefn(
            IN_STOCK_RATIO_YEAR_MONTH_STOCK_DAYS_LABEL_DEFN,
            pl.Int64,
            ["year", "month"],
        ),
    ],
)
