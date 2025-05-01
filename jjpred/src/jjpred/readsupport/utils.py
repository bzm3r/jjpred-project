"""Utility functions used commonly in :py:mod:`readsupport`."""

from __future__ import annotations
from collections.abc import Sequence

import polars as pl

from jjpred.channel import Channel
from jjpred.columns import Column
from jjpred.structlike import MemberType
from jjpred.utils.datetime import Date


def cast_standard(
    standard_dfs: list[pl.DataFrame],
    target_df: pl.DataFrame,
    use_dtype_of: dict[str, str] = {},
    strict: bool = True,
) -> pl.DataFrame:
    """Cast columns of a given target dataframe to the data types of the
      matching column in one of the "standard" dataframes.

    :param standard_dfs: Data types from these dataframes will be used to
        re-cast the data types of the target dataframe.
    :param target_df: The target dataframe whose columns should be recast.
    :param use_dtype_of: Sometimes column names do not match, in which case you
        can specify a mapping from (some, not necessarily all) column names
        from the target dataframe (key) to the column names from the target
        dataframe.
    :param strict: Whether casting should be done strictly (if not,
        :py:class:`pl.Null` will be inserted for failed casts).
    :return: The recasted target dataframe.
    """
    dtypes = {}
    use_dtype_of = {k: k for k in target_df.columns} | use_dtype_of

    for std_df in standard_dfs:
        dtypes |= {
            k: std_df[c].dtype for k, c in use_dtype_of.items() if c in std_df
        }

    return target_df.cast(dtypes, strict=strict)


NA_FBA_SHEET = "NA FBA Refill and Recall"
"""Name of the sheet in the FBA review main program Excel file which contains
the bulk of the information relevant to FBA review."""


def unpivot_dates(
    df: pl.DataFrame,
    id_cols: Sequence[Column | str],
    data_cols: Sequence[Column | str],
    value_name: str,
) -> pl.DataFrame:
    """Dates for historical sales information are given in the column headers.
    We need to "unpivot" (rotate) them along with the sales information they are
    associated with."""
    df = df.unpivot(
        index=[str(c) for c in id_cols],
        variable_name="date",
        value_name=value_name,
    )

    date_raw = pl.Series(
        "date_str", [str(c) for c in data_cols], dtype=pl.String()
    ).unique()
    date_parsed = pl.Series(
        "date_parsed",
        [Date.from_datelike(x).date for x in date_raw],
        dtype=pl.Date,
    )
    unique_dates = pl.DataFrame([date_raw, date_parsed])
    raw_to_parsed = dict(unique_dates.rows())

    # months_to_int = dict([(x, ix + 1) for ix, x in enumerate(MONTHS_LIST)])
    # years_to_int = dict([(x, int(x)) for x in unique_dates["year"]])

    df = df.with_columns(
        pl.col("date").replace(raw_to_parsed, return_dtype=pl.Date)
    )

    return df


def parse_channels(df: pl.DataFrame) -> pl.DataFrame:
    """Parse raw string channels in dataframes, and interpret them as
    :py:class:`Channel` objects."""
    unique_channels = df.select("channel").unique()
    unique_channels = unique_channels.with_columns(
        pl.col("channel")
        .map_elements(
            Channel.map_polars,
            return_dtype=Channel.intermediate_polars_type_struct(),
        )
        .alias("struct_channel")
    ).unnest("struct_channel")

    # country_issue = unique_channels.filter(pl.col.country_flag.is_null())
    # if len(country_issue) > 0:
    #     print(f"trouble parsing channels: \n{country_issue}")

    for c in Channel.members(MemberType.PRIMARY):
        if unique_channels[c].null_count() > 0:
            if c not in ["country_flag"]:
                raise ValueError(
                    "ERROR: found incorrectly parsed channels: \n"
                    f"{unique_channels.filter(pl.col(c).is_null())}"
                )
            # else:
            #     print(
            #         f"WARNING: issue parsing channels ({c}), replacing with 0: \n"
            #         f"{unique_channels.filter(pl.col(c).is_null())}"
            #     )
            #     unique_channels = unique_channels.with_columns(
            #         pl.when(pl.col(c).is_null())
            #         .then(0)
            #         .otherwise(pl.col(c))
            #         .alias(c)
            #     )

    unique_channels = unique_channels.cast(Channel.polars_type_dict())  # type: ignore
    unique_channels = unique_channels.rename(
        {"channel": "raw_channel"}
    ).with_columns(
        channel=pl.struct(Channel.members()).map_elements(
            Channel.map_polars_struct_to_string, return_dtype=pl.String()
        )
    )

    # for c in Channel.members(MemberType.PRIMARY):
    #     if unique_channels[c].dtype == pl.String():
    #         pl_enum = pl.Enum(pl.Series(unique_channels[c].unique()))
    #         unique_channels = unique_channels.with_columns(
    #             pl.col(c).cast(pl_enum)
    #         )
    #     elif c == "country_flag":
    #         pl.col(c).cast(PolarsCountryFlagType)

    if "raw_channel" not in df.columns and "channel" in df.columns:
        df = df.rename({"channel": "raw_channel"})

    df = df.join(
        unique_channels,
        on="raw_channel",
        validate="m:1",
        nulls_equal=True,
    )

    return df
