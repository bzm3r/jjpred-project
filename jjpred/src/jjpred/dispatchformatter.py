"""Format a dispatch calculation result into an output format expected by
NetSuite."""

import polars as pl
from jjpred.countryflags import CountryFlags
from jjpred.utils.datetime import Date, DateLike


def determine_season(month: int) -> str:
    """Determine the season part of the string placed in NetSuite-style dispatch
    file's ``SEASON`` column, which typically has entries like ``"24F"`` or
    ``"25S"``.

    ``F`` tag is given for March to August (inclusive).

    ``S`` tag is given for September to February (inclusive)."""
    if 3 <= month and month < 9:
        return "S"
    else:
        return "F"


def format_fba_dispatch_for_netsuite(
    analysis_date: DateLike,
    dispatch_results: pl.DataFrame,
    country_flag: CountryFlags,
    extra_cols: list[str] = [],
) -> pl.DataFrame:
    """Format a dispatch result dataframe as a NetSuite-style output file."""

    country_str = country_flag.try_to_string()
    # we expect that there is some country
    if country_str is None:
        raise ValueError(
            f"No logic for handling {country_flag=} with "
            f"string representation {country_str=}"
        )
    # for FBA dispatch, the country_flag should indicate a single country rather
    # than a group of countries
    assert len(country_str.split("|")) == 1, country_str.split("|")

    date: Date = Date.from_datelike(analysis_date)

    assert dispatch_results["dispatch"].dtype == pl.Int64()
    formatted_results = (
        dispatch_results.filter(pl.col.country_flag == int(country_flag))
        .select("sku", "fba_sku", "dispatch", *extra_cols)
        .with_columns(pl.col.dispatch.cast(pl.UInt64()))
        .rename({"sku": "ITEM", "dispatch": "Quantity", "fba_sku": "FBA SKUs"})
        .with_columns(
            pl.lit(date.format_as(r"%m/%d/%Y")).alias("Date"),
            pl.lit("FBA Replenishment").alias("TO TYPE"),
            pl.lit(
                f"{date.format_as(r'%y')}{determine_season(date.month)}"
            ).alias("SEASON"),
            pl.lit("WH-SURREY").alias("FROM WAREHOUSE"),
            pl.lit(f"WH-AMZ : FBA-{country_str}").alias("TO WAREHOUSE"),
            pl.lit(f"TO-FBA{country_str}{date.format_as(r'%y%m%d')}").alias(
                "REF NO"
            ),
            pl.lit("").alias("MEMO"),
            pl.lit("Gloria Li").alias("ORDER PLACED BY"),
        )
        .select(
            "Date",
            "TO TYPE",
            "SEASON",
            "FROM WAREHOUSE",
            "TO WAREHOUSE",
            "ITEM",
            "Quantity",
            "REF NO",
            "MEMO",
            "ORDER PLACED BY",
            "FBA SKUs",
            *extra_cols,
        )
    )

    return formatted_results


def format_jjweb_dispatch_for_netsuite(
    analysis_date: DateLike,
    dispatch_results: pl.DataFrame,
    country_flag: CountryFlags,
    extra_cols: list[str] = [],
) -> pl.DataFrame:
    """Format a dispatch result dataframe as a NetSuite-style output file."""

    country_str = country_flag.try_to_string()
    # we expect that there is some country
    if country_str is None:
        raise ValueError(
            f"No logic for handling {country_flag=} with "
            f"string representation {country_str=}"
        )
    # for FBA dispatch, the country_flag should indicate a single country rather
    # than a group of countries
    assert len(country_str.split("|")) == 1, country_str.split("|")

    date: Date = Date.from_datelike(analysis_date)

    assert dispatch_results["dispatch"].dtype == pl.Int64()
    formatted_results = (
        dispatch_results.filter(pl.col.country_flag == int(country_flag))
        .select("sku", "dispatch", *extra_cols)
        .with_columns(pl.col.dispatch.cast(pl.UInt64()))
        .rename({"sku": "ITEM", "dispatch": "Quantity"})
        .with_columns(
            pl.lit(date.format_as(r"%m/%d/%Y")).alias("Date"),
            pl.lit("FBA Replenishment").alias("TO TYPE"),
            pl.lit(
                f"{date.format_as(r'%y')}{determine_season(date.month)}"
            ).alias("SEASON"),
            pl.lit("WH-SURREY").alias("FROM WAREHOUSE"),
            pl.lit(f"WH-AMZ : FBA-{country_str}").alias("TO WAREHOUSE"),
            pl.lit(f"TO-FBA{country_str}{date.format_as(r'%y%m%d')}").alias(
                "REF NO"
            ),
            pl.lit("").alias("MEMO"),
            pl.lit("Gloria Li").alias("ORDER PLACED BY"),
        )
        .select(
            "Date",
            "TO TYPE",
            "SEASON",
            "FROM WAREHOUSE",
            "TO WAREHOUSE",
            "ITEM",
            "Quantity",
            "REF NO",
            "MEMO",
            "ORDER PLACED BY",
            *extra_cols,
        )
    )

    return formatted_results
