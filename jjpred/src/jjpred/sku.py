"""Representation of SKU information for a product. It typically consists of:
various SKUs (e.g. merchant sku, adjusted merchant sku) and the category, print,
and size information, along with "``sku_remainder``": extra SKU information such
as ``V1`` etc."""

from __future__ import annotations

from dataclasses import field
import re
from typing import Self
import polars as pl
from jjpred.parse.patternmatch import (
    CompiledPattern,
    CompiledMatchSkip,
    ReMatchCondition,
    ReMatchResult,
    ReMatcher,
    StringPattern,
)
from jjpred.structlike import FieldMeta, MemberType, StructLike
from jjpred.utils.typ import do_nothing

type Category = str
type Print = str
type Size = str
type SkuRemainder = str

SKU_KEYS = ["sku", "category", "print", "size", "sku_remainder"]
SKU_REMAINDER = [x for x in SKU_KEYS if x not in ["sku", "category"]]
SKU_PART_PATTERN = r"[^-]+"
CATEGORY_PATTERN = StringPattern(SKU_PART_PATTERN).named("category")
REMAINDER_PATTERN = [
    StringPattern("-")
    .concatenate(StringPattern(SKU_PART_PATTERN).named(x))
    .no_capture()
    .optional()
    for x in [k for k in SKU_REMAINDER]
]
SKU_PATTERN = CATEGORY_PATTERN.concatenate(*REMAINDER_PATTERN).named("sku")
SKU_MATCHER = ReMatcher(
    "sku",
    CompiledMatchSkip(
        CompiledPattern(SKU_PATTERN, ReMatchResult, flags=re.IGNORECASE)
    ),
    ReMatchCondition.DeepAll,
)


class Sku(StructLike, matcher=SKU_MATCHER, joiner="-"):
    sku: str = field(
        default_factory=str,
        compare=False,
        metadata=FieldMeta(MemberType.META, do_nothing, pl.String()),
    )
    category: Category = field(
        compare=False,
        metadata=FieldMeta(MemberType.PRIMARY, do_nothing, pl.String()),
    )
    print: Print = field(
        default_factory=str,
        compare=False,
        metadata=FieldMeta(MemberType.PRIMARY, do_nothing, pl.String()),
    )
    size: Size = field(
        default_factory=str,
        compare=False,
        metadata=FieldMeta(MemberType.PRIMARY, do_nothing, pl.String()),
    )
    sku_remainder: SkuRemainder = field(
        default_factory=str,
        metadata=FieldMeta(MemberType.SECONDARY, do_nothing, pl.String()),
    )

    @classmethod
    def from_dict(cls, x: dict[str, str]) -> Self:
        return cls(
            **(
                cls.field_defaults
                | {
                    k: x[k]
                    for k in cls.fields_by_cutoff[MemberType.SECONDARY]
                    if x.get(k) is not None
                }
            )
        )


SKU_CATEGORY = ["category"]
SKU_PRINT = ["print"]
SKU_SIZE = ["size"]
SKU_REMAINDER = ["sku_remainder"]
SKU_CATEGORY_PRINT = SKU_CATEGORY + SKU_PRINT
SKU_CATEGORY_PRINT_SIZE = SKU_CATEGORY_PRINT + SKU_SIZE
SKU_CATEGORY_PRINT_SIZE_REMAINDER = SKU_CATEGORY_PRINT_SIZE + SKU_REMAINDER
SKU_PRINT_SIZE = SKU_PRINT + SKU_SIZE
