"""An object which is usually given as a string in input data, but actually has
internal structure (for example, a SKU has :code:`category-print-size-remainder`
structure, while a Channel has :code:`platform-country_flag-mode structure`
structure)."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import _MISSING_TYPE, Field, dataclass, field
import dataclasses
from enum import Enum, auto
from functools import total_ordering
from typing import (
    Any,
    Hashable,
    Protocol,
    Self,
    runtime_checkable,
)
import polars as pl

from jjpred.parse.patternmatch import PatternMatcher
from jjpred.utils.typ import as_type


@total_ordering
class MemberType(Enum):
    META = auto()
    SECONDARY = auto()
    PRIMARY = auto()

    def __lt__(self, other: Self) -> bool:
        return self.value < other.value

    @classmethod
    def after(cls, priority: Self) -> Self:
        for x in cls:
            if x > priority:
                return x

        return priority


def normalize_missing[T](
    value: _MISSING_TYPE | T | None,
) -> T | None:
    if isinstance(value, _MISSING_TYPE):
        return None
    else:
        return value


@dataclass
class FieldMeta[T](Mapping):
    priority: MemberType
    polars_parser: Callable[[Any], T]
    polars_dtype: pl.DataType
    intermediate_polars_dtype: pl.DataType | None = field(default=None)
    default_value: Any | None = field(default=None)

    def __getitem__(self, key: Any) -> Any:
        return self.__dict__[key]

    def __iter__(self) -> Iterator:
        return self.__dict__.__iter__()

    def __len__(self) -> int:
        return self.__dict__.__len__()

    def with_default(
        self,
        default_value: _MISSING_TYPE | T | None,
        default_factory: _MISSING_TYPE | Callable[[], T] | None,
    ) -> Self:
        default_value = normalize_missing(default_value)
        default_factory = normalize_missing(default_factory)

        if default_value is not None and default_factory is not None:
            raise ValueError(
                f"{default_value=} and {default_factory=}: only one expected"
            )
        else:
            if default_value is not None:
                self.default_value = default_value
            elif default_factory is not None:
                self.default_value = default_factory()
            else:
                self.default_value = None

        return self


@dataclass(init=False, repr=False, eq=False)
@runtime_checkable
class StructLike(Hashable, Protocol):
    matcher: PatternMatcher = field(init=False)
    joiner: str = field(init=False)
    fields_by_priority: dict[MemberType, list[str]] = field(
        init=False, default_factory=dict
    )
    fields_by_cutoff: dict[MemberType, list[str]] = field(
        init=False, default_factory=dict
    )
    field_dtypes: dict[str, pl.DataType] = field(
        init=False, default_factory=dict
    )
    field_intermediate_dtypes: dict[str, pl.DataType] = field(
        init=False, default_factory=dict
    )
    field_parsers: dict[str, Callable[[Any], Any]] = field(
        init=False, default_factory=dict
    )
    field_defaults: dict[str, Any] = field(init=False, default_factory=dict)
    field_metadata: dict[str, FieldMeta] = field(
        init=False, default_factory=dict
    )
    str_reprs: dict[MemberType, str] = field(init=False, default_factory=dict)

    def __init_subclass__(
        cls,
        matcher: PatternMatcher,
        joiner: str,
    ) -> None:
        structlike_fields = list(cls.__dataclass_fields__.keys())

        dataclasses.dataclass(kw_only=True, eq=False, repr=False)(cls)
        subclass_fields = list(
            [
                x
                for x in cls.__dataclass_fields__.keys()
                if x not in structlike_fields
            ]
        )

        cls.field_metadata: dict[str, FieldMeta] = {}
        for x in subclass_fields:
            field_meta = as_type(cls.__dataclass_fields__[x], Field)
            metadata = as_type(field_meta.metadata, FieldMeta).with_default(
                field_meta.default, field_meta.default_factory
            )
            cls.field_metadata[x] = metadata

        cls.matcher = matcher
        cls.joiner = joiner

        cls.fields_by_priority = {k: list() for k in MemberType}
        cls.fields_by_cutoff = {k: list() for k in MemberType}
        for name, field_meta in cls.field_metadata.items():
            cls.fields_by_priority[field_meta.priority].append(name)
            for priority in MemberType:
                if not (field_meta.priority < priority):
                    cls.fields_by_cutoff[priority].append(name)

        assert len(cls.fields_by_priority[MemberType.META]) <= 1

        cls.field_dtypes = {
            k: cls.field_metadata[k].polars_dtype
            for k in cls.fields_by_cutoff[MemberType.META]
        }

        cls.field_intermediate_dtypes = {  # type: ignore
            k: cls.field_metadata[k].intermediate_polars_dtype
            if cls.field_metadata[k].intermediate_polars_dtype is not None
            else cls.field_metadata[k].polars_dtype
            for k in cls.fields_by_cutoff[MemberType.META]
        }

        cls.field_parsers = {
            k: cls.field_metadata[k].polars_parser
            for k in cls.fields_by_cutoff[MemberType.META]
        }

        cls.field_defaults = {}
        for k in cls.fields_by_cutoff[MemberType.SECONDARY]:
            v = cls.field_metadata[k].default_value
            if v is not None:
                cls.field_defaults[k] = v

        cls.str_reprs = {}

    @classmethod
    def normalize_keys(
        cls,
        keys: set[str] | list[str] | None,
        at_least: MemberType = MemberType.SECONDARY,
    ) -> list[str]:
        if keys is None or len(keys) == 0:
            return cls.fields_by_cutoff[at_least]
        elif isinstance(keys, set):
            return sorted(keys, key=cls.fields_by_cutoff[at_least].index)
        else:
            return keys

    def get_field_from_dict(
        self,
        k: str,
        d: dict[str, Any],
    ) -> Any:
        if k in d.keys():
            return k
        else:
            if k not in self.__class__.field_defaults.keys():
                raise ValueError(
                    f"Requires value for {k}, but {d} does not contain it."
                )
            else:
                return self.__class__.field_defaults[k]

    def is_default(self, k: str, v: Any) -> bool:
        if k in self.__class__.field_defaults.keys():
            field_default = self.__class__.field_defaults[k]
            return (v is None and field_default is None) or (
                v == field_default
            )
        else:
            return False

    def as_dict(
        self,
        keys: set[str] | list[str] | None = None,
        remove_defaults: bool = True,
    ) -> dict[str, object]:
        keys = self.__class__.normalize_keys(keys)

        result = dict()
        for k in keys:
            if k in self.__dict__.keys():
                v = self.__dict__[k]
                if not (remove_defaults and self.is_default(k, v)):
                    v = self.__class__.field_parsers[k](v)
                    result[k] = v

        return result

    def to_columns(
        self,
        keys: set[str] | list[str] | None = None,
        remove_defaults: bool = True,
    ) -> dict[str, pl.Expr]:
        return {
            k: pl.lit(v)
            for k, v in self.as_dict(
                keys=keys, remove_defaults=remove_defaults
            ).items()
        }

    @classmethod
    def from_dict(cls, x: dict[str, Any]) -> Self:
        raise NotImplementedError()

    @classmethod
    def from_str(cls, x: str) -> Self:
        if cls.matcher and (result := cls.matcher.apply(x)):
            return cls.from_dict(result.data)
        else:
            raise ValueError(f"Cannot parse {x} as {cls.__name__}")

    @classmethod
    def try_from_str(cls, x: str) -> Self | None:
        try:
            return cls.from_str(x)
        except ValueError:
            return None

    @classmethod
    def parse(cls, x: str | dict | Self | None) -> Self:
        if x is not None:
            if isinstance(x, cls):
                return x
            elif isinstance(x, str):
                return cls.from_str(x.strip())
            elif isinstance(x, dict):
                return cls.from_dict(x)

        raise ValueError(f"Cannot parse {x=} as {cls.__name__}")

    @classmethod
    def default_value(cls) -> Self | None:
        return None

    @classmethod
    def map_polars(
        cls, x: str | dict | Self | None, keys: list[str] = list()
    ) -> dict[str, Any] | None:
        keys = cls.normalize_keys(keys)

        if isinstance(x, str | dict):
            x = cls.parse(x)

        if x is None:
            return cls.field_defaults

        if x is not None:
            return x.as_dict(keys, remove_defaults=False)

    @classmethod
    def intermediate_polars_type_dict(
        cls, keys: list[str] = list()
    ) -> dict[str, pl.DataType]:
        keys = cls.normalize_keys(keys)
        return {
            k: v for k, v in cls.field_intermediate_dtypes.items() if k in keys
        }

    @classmethod
    def polars_type_dict(
        cls, keys: list[str] = list()
    ) -> dict[str, pl.DataType]:
        keys = cls.normalize_keys(keys)
        return {k: v for k, v in cls.field_dtypes.items() if k in keys}

    @classmethod
    def intermediate_polars_type_struct(
        cls, keys: list[str] = list()
    ) -> pl.Struct:
        return pl.Struct(cls.intermediate_polars_type_dict())

    @classmethod
    def polars_type_struct(cls, keys: list[str] = list()) -> pl.Struct:
        return pl.Struct(cls.polars_type_dict())

    @classmethod
    def members(cls, at_least: MemberType = MemberType.SECONDARY) -> list[str]:
        return list(cls.fields_by_cutoff[at_least])

    def raw_str(self) -> str:
        result = (
            f"({"".join(self.__class__.fields_by_priority[MemberType.META])})"
        )
        if len(result) > 0:
            return result
        else:
            return None.__repr__()

    def str_repr(self, at_least: MemberType) -> str:
        if result := self.str_reprs.get(at_least):
            return result
        result = ""

        if at_least >= MemberType.META:
            parts = []
            for x in self.members(
                max(MemberType.after(MemberType.META), at_least)
            ):
                part = str(self.__dict__.get(x))
                if part is not None and len(part) > 0:
                    parts.append(part)

            result = self.joiner.join(parts)

        if at_least == MemberType.META:
            result = self.raw_str() + " -> " + result

        self.str_reprs[at_least] = result
        return result

    def __str__(self) -> str:
        return self.str_repr(MemberType.SECONDARY)

    def __repr__(self) -> str:
        return self.str_repr(MemberType.SECONDARY)

    def _is_valid_operand(self, other: object):
        return isinstance(other, self.__class__) or isinstance(other, str)

    def __lt__(self, other: object) -> bool:
        return str(self) < str(other)

    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return self.str_repr(MemberType.SECONDARY).__hash__()
