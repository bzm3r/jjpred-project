"""Utilities useful when reading or writing Excel files."""

import polars as pl
from jjpred.countryflags import CountryFlags
from jjpred.utils.polars import get_columns_in_df
from jjpred.utils.typ import normalize_as_list


def map_pause_plan_int(x: int) -> str:
    """Convert country flag codes representing a pause plan into a string."""
    if x == 0:
        return "All Active"
    else:
        return str(CountryFlags.from_int(x))


def map_pause_plan(x: int | list[int] | pl.Series) -> str:
    """Convert country flag codes representing a pause plan into a string."""
    if isinstance(x, pl.Series):
        x = list(x)
    assert isinstance(x, int | list)
    return ",".join([map_pause_plan_int(y) for y in normalize_as_list(x)])


def map_country_flag_int(x: int) -> str:
    """Convert country flag codes representing a pause plan into a string."""
    if x == 0:
        return "NO_COUNTRY"
    else:
        return str(CountryFlags.from_int(x))


def map_country_flag(x: int | list[int]) -> str:
    """Convert country flag codes representing a pause plan into a string."""
    if isinstance(x, pl.Series):
        x = list(x)
    assert isinstance(x, int | list)
    return ",".join([map_country_flag_int(y) for y in normalize_as_list(x)])


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


def country_flag_to_string(x: int) -> str:
    """Convert country flag codes into a string representation."""
    if x == 0:
        return "All Active"
    else:
        return str(CountryFlags.from_int(x))


def convert_df_for_excel(df: pl.DataFrame) -> pl.DataFrame:
    """Convert a dataframe for representation as an Excel file.

    We have to convert:
     * lists of ``polars.Date``s or ``polars.Enum`` elements into
       comma-separated string lists.
     * country flag codes into string representations
    """

    with_converted_lists = df.with_columns(
        pl.col(x).list.eval(pl.element().cast(pl.String())).list.join(", ")
        for x in get_columns_in_df(df, ["current_period", "referred_by"])
    )

    return normalize_pause_plan_and_country_flags_for_excel(
        with_converted_lists
    )
