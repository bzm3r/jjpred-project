"""Utilites for dealing with Polars dataframes and datatypes."""

from __future__ import annotations
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum, IntFlag
from functools import reduce
import sys
from typing import (
    Any,
    Literal,
    Self,
)
import polars as pl
import polars.selectors as cs

from jjpred.parse.patternmatch import StringPattern
from jjpred.structlike import StructLike
from jjpred.utils.typ import (
    ScalarOrList,
    as_polars_type,
    create_assert_result,
    normalize_as_list,
    normalize_optional,
    normalize_scalar_or_list_of_sets,
)


def get_columns_in_df(df: pl.DataFrame, columns: list[str]) -> list[str]:
    return [x for x in df.columns if x in columns]


class FilterStructs:
    """Filter a dataframe using struct-like objects which have multiple parts
    (e.g. channels, or SKUs).

    Based on this `SO question/answer <https://stackoverflow.com/questions/72546690/lightweight-syntax-for-filtering-a-polars-dataframe-on-a-multi-column-key>`_.
    """

    __structs__: dict[str, list[StructLike]]

    def __init__(
        self,
        *filters: ScalarOrList[StructLike | set[StructLike]] | Self | None,
    ):
        self.__structs__ = {}
        for f in filters:
            if not isinstance(f, FilterStructs):
                f = normalize_scalar_or_list_of_sets(f)
                for struct_like in f:
                    k = struct_like.__class__.__name__
                    if self.__structs__.get(k):
                        self.__structs__[k].append(struct_like)
                    else:
                        self.__structs__[k] = [struct_like]
            else:
                self.__structs__ |= f.__structs__

    def iter(self) -> Iterable[list[StructLike]]:
        for v in self.__structs__.values():
            yield v


# see: https://stackoverflow.com/questions/72546690/lightweight-syntax-for-filtering-a-polars-dataframe-on-a-multi-column-key
def struct_filter(
    df: pl.DataFrame,
    *filters: ScalarOrList[StructLike | set[StructLike]]
    | FilterStructs
    | None,
) -> pl.DataFrame:
    """Filter a dataframe using struct-like objects."""
    for filter_list in FilterStructs(*filters).iter():
        if len(filter_list) > 0 and df.shape[0] > 0:
            filter_keys = filter_list[0].as_dict(remove_defaults=True).keys()
            for other in filter_list[1:]:
                filter_keys = [
                    x for x in filter_keys if x in other.as_dict().keys()
                ]
            relevant_keys = [x for x in filter_keys if x in df.columns]
            if len(relevant_keys) > 0:
                filter_dtype = pl.Struct(
                    {k: df[k].dtype for k in relevant_keys}
                )
                df = df.filter(
                    pl.struct(relevant_keys).is_in(
                        pl.Series(
                            [s.as_dict(relevant_keys) for s in filter_list],
                            dtype=filter_dtype,
                        )
                    )
                )
    return df


def binary_partition(
    df: pl.DataFrame,
    *predicate_expr: pl.Expr,
) -> dict[bool, pl.DataFrame]:
    """Partition a dataframe based on a predicate (True/False) Polars
    expression."""
    groups = {k: pl.DataFrame(schema=df.schema) for k in [True, False]}
    for key, group in df.group_by(*predicate_expr):
        assert all(x is not None and isinstance(x, bool) for x in key), print(
            key, group
        )
        groups[all(key)] = group
    return groups


def check_is_df(x: Any) -> pl.DataFrame:
    assert isinstance(x, pl.DataFrame)
    return x


def enum_extend_vstack(
    df: pl.DataFrame | None, other_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Vertically stack two dataframes, ensuring that the enum types for their
    columns with an enum types are extended appropriately so that they can be
    vertically stacked.
    """
    if df is None:
        return other_df
    else:
        assert sorted(df.columns) == sorted(other_df.columns), (
            sorted(df.columns),
            sorted(other_df.columns),
        )

        for column in df.columns:
            if isinstance(df[column].dtype, pl.Enum):
                categories = as_polars_type(
                    df[column].dtype, pl.Enum
                ).categories
                other_categories = as_polars_type(
                    other_df[column].dtype, pl.Enum
                ).categories
                if (not (len(categories) == len(other_categories))) or (
                    not categories.eq(other_categories).all()
                ):
                    combined_categories = (
                        categories.extend(other_categories).unique().sort()
                    )
                    df = df.cast({column: pl.Enum(combined_categories)})
                    other_df = other_df.cast(
                        {column: pl.Enum(combined_categories)}
                    )

        return df.vstack(other_df.select(df.columns))


def concat_enum_extend_vstack(dfs: list[pl.DataFrame]) -> pl.DataFrame | None:
    return reduce(enum_extend_vstack, dfs, None)


def concat_enum_extend_vstack_strict(
    dfs: list[pl.DataFrame],
) -> pl.DataFrame:
    result = reduce(enum_extend_vstack, dfs, None)
    assert result is not None

    return result.rechunk()


def binary_partition_weak(
    df: pl.DataFrame,
    *predicate_expr: pl.Expr,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Partition a dataframe based on a predicate (True/False) Polars
    expression.

    Returns a pair of dataframes: the left dataframe is the elements
    where the expression evaluates true, the right dataframe is the elements
    where the expression evaluates False or None."""
    key_list: list[Literal[True, "other"]] = [True, "other"]
    group_list: dict[Literal[True, "other"], list[pl.DataFrame]] = {
        k: [] for k in key_list
    }
    for key, group in df.group_by(*predicate_expr):
        if all(x is not None and isinstance(x, bool) for x in key):
            condition = all(key)
            if condition:
                group_list[condition].append(group)
            else:
                group_list["other"].append(group)
        else:
            group_list["other"].append(group)

    groups = {}
    for key, dfs in group_list.items():
        concat_df = concat_enum_extend_vstack(dfs)
        if concat_df is None:
            concat_df = pl.DataFrame(schema=df.schema)
        groups[key] = concat_df

    return groups[True], groups["other"]


def binary_partition_strict(
    df: pl.DataFrame,
    *predicate_expr: pl.Expr,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Partition a dataframe based on a predicate (True/False) Polars
    expression. Raises errors if the dataframe cannot be "purely" partitioned
    into parts that either evaluate True/False, but instead there is a part that
    also evalues to ``None`` (i.e. predicate expression given sometimes
    encounters missing data)."""
    groups = binary_partition(df, *predicate_expr)
    assert len(groups.keys()) == 2
    assert True in groups.keys() and False in groups.keys()
    return groups[True], groups[False]


def scalar_as_series[T, U](
    x: T,
    converter: Callable[[T], U] | None = None,
    dtype: pl.DataType | None = None,
) -> pl.Series:
    """Convert a scalar value into a Polars Series."""

    def f(x: T) -> T | U:
        if isinstance(converter, Callable):
            return converter(x)
        return x

    return pl.Series([f(x)], dtype=dtype)


class SumTypeLike:
    """Building block for representing objects which can be represented using a
    :py:class:`pl.Enum`."""

    def try_to_string(self) -> str | None:
        raise NotImplementedError()

    def polars_lit(self) -> pl.Expr | None:
        """Get a Polars literal representing this particular value."""
        raise NotImplementedError()

    @classmethod
    def combinations(cls) -> list[Self]:
        raise NotImplementedError()

    @classmethod
    def polars_type(cls) -> Any:
        raise NotImplementedError()

    @classmethod
    def try_from_str(cls, string: str | None) -> Self | None:
        try:
            if string:
                return cls.from_str(string)
        except ValueError:
            ...
        return None

    def string_pattern(self) -> StringPattern:
        return StringPattern(rf"{str(self)}")

    @classmethod
    def from_str(cls, string: str) -> Self:
        lower_string = string.lower()
        if string:
            for variant in cls.combinations():
                name = variant.try_to_string()
                if name is not None:
                    if name.lower() == lower_string:
                        return variant

        raise ValueError(f"Cannot parse {string} as {cls.__qualname__}.")

    def __str__(self) -> str:
        if self.try_to_string() is not None:
            return f"{self.try_to_string()}"
        else:
            return f"NO_{self.__class__.__qualname__.upper()}"

    def __repr__(self) -> str:
        return str(self)


class EnumLike(SumTypeLike, Enum):
    """Building block for objects that can be represented using
    :py:class:`pl.Enum`."""

    def polars_lit(self) -> pl.Expr | None:
        if self.try_to_string() is not None:
            return pl.lit(self.name, dtype=self.polars_type())
        else:
            return None

    def try_to_string(self) -> str | None:
        return self.name

    @classmethod
    def combinations(cls) -> list[Self]:
        return list(cls)

    @classmethod
    def polars_type(cls) -> pl.Enum:
        variant_labels = []
        for variant in cls:  # <--- this is where the static checker is unhappy
            variant_labels.append(variant.name)
        return pl.Enum(sorted(variant_labels))

    @classmethod
    def parse_as_polars_lit(cls, x: str | EnumLike) -> pl.Expr:
        if isinstance(x, cls):
            this = x
        else:
            assert isinstance(x, str)
            this = cls.from_str(x)

        return pl.lit(this.name, dtype=cls.polars_type())


class IntFlagLike(SumTypeLike, IntFlag):
    """Building block for :py:class:`IntFlag` objects which are to be
    represented in Polars as a :py:class:`pl.Int64`."""

    def polars_lit(self) -> pl.Expr | None:
        if self.value is not None:
            return pl.lit(self.value, dtype=pl.Int64())
        else:
            return None

    def try_to_string(self) -> str | None:
        return self.name

    @classmethod
    def try_from_str(cls, string: str | None) -> Self | None:
        try:
            if string:
                return cls.from_str(string)
            else:
                return cls(0)
        except ValueError:
            return None

    @classmethod
    def max_int(cls) -> int:
        return 1 << len(cls)

    @classmethod
    def from_int(cls, x: int) -> Self:
        if 0 <= x <= cls.max_int():
            return cls(x)
        else:
            raise ValueError(
                f"Cannot parse {x} as {cls.__qualname__}; "
                f"max_int = {cls.max_int()}"
            )

    @classmethod
    def parse(cls, x: int | str) -> Self:
        if isinstance(x, int):
            return cls.from_int(x)
        elif isinstance(x, str):
            return cls.from_str(x)
        else:
            raise ValueError(f"No logic to parse {x} as {cls.__qualname__}.")

    @classmethod
    def try_parse(cls, x: int | str | None) -> Self | None:
        if x is not None:
            return cls.parse(x)
        else:
            return None

    @classmethod
    def combinations(cls) -> list[Self]:
        combinations = []

        for i in range(cls.max_int()):
            combination = cls(0)
            for j in cls:
                if i & int(j):
                    combination |= j

            combinations.append(combination)

        return combinations

    @classmethod
    def polars_type(cls) -> pl.DataType:
        return pl.Int64()


def find_dupes(
    df: pl.DataFrame,
    id_cols: list[str],
    ignore_columns: list[str] | None = None,
    raise_error: bool = False,
) -> pl.DataFrame:
    """Find duplicates in a Polars dataframe."""
    ignore_columns = normalize_optional(ignore_columns, [])
    non_id_cols = cs.expand_selector(df, cs.exclude(id_cols, *ignore_columns))
    dupes = (
        df.with_columns(is_dupe=pl.struct(id_cols).is_duplicated())
        .filter(pl.col("is_dupe"))
        .drop("is_dupe")
        .group_by(id_cols)
        .agg(pl.col(x) for x in non_id_cols)
    )

    if raise_error and len(dupes) > 0:
        sys.displayhook(dupes.sort([x for x in id_cols if x in dupes.columns]))
        raise ValueError("found dupes!")
    return dupes


def sanitize_excel_extraction(df: pl.DataFrame) -> pl.DataFrame:
    """Sanitize a dataframe extracted from an Excel file by removing all rows
    which are completely :py:class:`pl.Null` and removing all leading and
    trailing whitespace from string-valued columns."""
    df_columns = df.columns

    # remove all completely null rows
    df = df.filter(~pl.all_horizontal(pl.all().is_null()))

    # remove leading and trailing whitespace
    df = (
        df.select(cs.exclude(cs.string()))
        .with_columns(
            df.select(cs.string()).with_columns(pl.all().str.strip_chars())
        )
        .select(df_columns)
    )

    return df


@dataclass
class OverrideLeft:
    """Coalesce mode: override the values in the left dataframe using the values
    in the right dataframe when encountering columns with the same name in two
    dataframes to be joined."""

    index: list[str]
    """The column names that act as the "index" of the join: these are the keys
    on which a join is performed."""


@dataclass
class NoOverride:
    """Coalesce mode: do not override values when encountering columns with the
    same name in two dataframes to be joined."""

    ...


def join_and_coalesce(
    left_df: pl.DataFrame,
    right_df: pl.DataFrame,
    coalesce: OverrideLeft | NoOverride,
    nulls_equal: bool = False,
    dupe_check_index: list[str] | None = None,
    raise_dupe_err: bool = True,
):
    """Join two dataframes, and coalesce their values depending on the coalesce
    mode provided."""
    if isinstance(coalesce, OverrideLeft):
        index = coalesce.index
    else:
        index = [x for x in left_df.columns if x in right_df.columns]

    left_index = [x for x in left_df.columns if x in index]

    assert len(left_index) > 0, create_assert_result(
        left_df_cols=left_df.columns, index_cols=index
    )
    shared_index = [x for x in right_df.columns if x in left_index]
    assert len(shared_index) > 0, create_assert_result(
        right_df_cols=right_df.columns, left_index_cols=left_index
    )
    left_data_cols = [x for x in left_df.columns if x not in left_index]
    right_data_cols = [x for x in right_df.columns if x not in shared_index]
    shared_data_cols = [x for x in right_data_cols if x in left_data_cols]
    unique_data_cols = [
        x
        for x in left_data_cols
        if not (x in right_data_cols or x in left_index)
    ] + [
        x
        for x in right_data_cols
        if not (x in shared_data_cols or x in shared_index)
    ]

    if isinstance(coalesce, OverrideLeft):
        joined_df = left_df.join(
            right_df, on=shared_index, how="full", nulls_equal=nulls_equal
        )
        coalesced_df = joined_df.with_columns(
            pl.coalesce(
                f"{x}_right",
                f"{x}",
            )
            for x in shared_index + shared_data_cols
        )
        processed_df = (
            coalesced_df.drop(f"{x}" for x in shared_index + shared_data_cols)
            .rename(
                {f"{x}_right": f"{x}" for x in shared_index + shared_data_cols}
            )
            .select(left_index + shared_data_cols + unique_data_cols)
        )
    elif isinstance(coalesce, NoOverride):
        a, b = pl.align_frames(
            left_df,
            right_df,
            on=shared_index + shared_data_cols,
        )
        processed_df = a.join(b, on=shared_index + shared_data_cols)
    else:
        raise ValueError(f"No logic to handle: {coalesce=}")

    if dupe_check_index is None and raise_dupe_err:
        dupe_check_index = shared_index

    if dupe_check_index:
        find_dupes(processed_df, dupe_check_index, raise_error=True)

    return processed_df


def vstack_to_unified(
    existing_df: pl.DataFrame | None, new_df: pl.DataFrame
) -> pl.DataFrame:
    """Concatenate Polars dataframe to an existing dataframe."""
    if existing_df is None or len(existing_df) == 0:
        return new_df
    else:
        return existing_df.vstack(
            new_df.select(existing_df.columns).cast(
                {k: existing_df[k].dtype for k in existing_df.columns}
            )
        )


def convert_dict_to_polars_df(
    input_dict: dict[Any, Any], key_column_name: str, value_column_name: str
) -> pl.DataFrame:
    return pl.from_dicts(
        [
            {key_column_name: k, value_column_name: v}
            for k, v in input_dict.items()
        ]
    )


def polars_integer(size: Literal[32, 64]) -> pl.Int32 | pl.Int64:
    if size == 32:
        return pl.Int32()
    else:
        return pl.Int64()


def polars_float(size: Literal[32, 64]) -> pl.Float32 | pl.Float64:
    if size == 32:
        return pl.Float32()
    else:
        return pl.Float64()


def extend_df_enum_type(
    df: pl.DataFrame, column: str, extensions: ScalarOrList[str]
) -> pl.DataFrame:
    """Extend the dataframe's channel column with required extensions."""
    enum_extensions = pl.Series(column, normalize_as_list(extensions))

    if column in df.columns:
        channel_dtype = as_polars_type(df[column].dtype, pl.Enum)
        channel_dtype = pl.Enum(
            channel_dtype.categories.extend(enum_extensions).unique().sort()
        )
        df = df.cast({column: channel_dtype})

    return df


def check_dfs_for_differences(
    df: pl.DataFrame,
    other_df: pl.DataFrame,
    index_cols: list[str],
    raise_error: bool = False,
) -> pl.DataFrame:
    all_columns = []

    assert sorted(df.columns) == sorted(other_df.columns), [
        x
        for x in sorted(list(set(df.columns).union(other_df.columns)))
        if x not in df or x not in other_df
    ]

    for x in df.columns:
        all_columns.append(x)
        all_columns.append(f"{x}_right")
        all_columns.append(f"{x}_same")

    check_df = (
        df.join(other_df, on=index_cols, how="full")
        .with_columns(
            pl.col(x).eq(pl.col(f"{x}_right")).alias(f"{x}_same")
            for x in df.columns
        )
        .select(all_columns)
    )

    check_df = check_df.with_columns(
        check_failed=pl.any_horizontal(
            [~pl.col(f"{x}_same") for x in df.columns]
        )
    )

    check_failed = check_df.filter(
        pl.col.check_failed,
    )

    if len(check_failed) == 0:
        print("check passed.")
    else:
        if raise_error:
            sys.displayhook(check_failed)
            raise ValueError("check failed!")

        print("check failed.")

    return check_failed


def display_rows(df: pl.DataFrame, rows: int | None = None) -> None:
    if rows is not None:
        pl.Config().set_tbl_rows(rows)
    sys.displayhook(df)
    pl.Config().restore_defaults()
