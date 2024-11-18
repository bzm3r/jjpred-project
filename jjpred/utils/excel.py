"""Utilities useful when reading or writing Excel files."""

import polars as pl
from jjpred.countryflags import CountryFlags
from jjpred.utils.polars import get_columns_in_df


def map_pause_plan(x: int) -> str:
    """Convert country flag codes representing a pause plan into a string."""
    if x == 0:
        return "All Active"
    else:
        return str(CountryFlags.from_int(x))


def map_country_flag(x: int) -> str:
    """Convert country flag codes representing a pause plan into a string."""
    if x == 0:
        return "NO_COUNTRY"
    else:
        return str(CountryFlags.from_int(x))


def normalize_pause_plan_and_country_flags_for_excel(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Convert pause plans and country flags from a dataframe into an human
    readable strings for Excel."""
    df = df.with_columns(
        pl.col(x).map_elements(map_pause_plan, return_dtype=pl.String())
        for x in get_columns_in_df(df, ["pause_plan"])
    )
    df = df.with_columns(
        pl.col(x).map_elements(map_country_flag, return_dtype=pl.String())
        for x in get_columns_in_df(df, ["country_flag"])
    )

    return df
