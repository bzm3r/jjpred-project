"""Functions used to help examine and plot sales trends related to size-related
sales differences in Excel files."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from typing import Literal
import polars as pl
import polars.selectors as cs
import altair as alt


from jjpred.channel import Channel
from jjpred.countryflags import CountryFlags
from jjpred.inputstrategy import TimePeriod
from jjpred.utils.datetime import Date, Month
from jjpred.utils.polars import struct_filter
from jjpred.utils.typ import ScalarOrList, normalize_as_list


type ChannelFilterKey = (
    Literal["Amazon CA"]
    | Literal["Amazon US"]
    | Literal["Amazon CA/US"]
    | Literal["janandjul.com"]
    | Literal["All CA/US Retail"]
    | Literal["Non-CA/US Retail"]
    | Literal["Amazon.co.uk"]
    | Literal["Amazon.de"]
    | Literal["Amazon.com.au"]
    | Literal["Amazon EU/UK"]
    | Literal["Wholesale"]
)

CHANNEL_FILTERS: dict[
    ChannelFilterKey, ScalarOrList[str | Channel] | pl.Expr
] = {
    "Amazon CA": ["Amazon CA"],
    "Amazon US": ["Amazon US"],
    "Amazon EU/UK": ["Amazon.co.uk", "Amazon.eu", "Amazon.uk", "Amazon.de"],
    "Amazon CA/US": ["Amazon.ca", "Amazon.com"],
    "All CA/US Retail": pl.col("country_flag")
    .and_(CountryFlags.CA | CountryFlags.US)
    .gt(0)
    & pl.col("mode").eq("RETAIL"),
    "janandjul.com": ["janandjul.com"],
    "Non-CA/US Retail": pl.col("country_flag")
    .and_(CountryFlags.CA | CountryFlags.US)
    .eq(0)
    & pl.col("mode").eq("RETAIL"),
    "Amazon.co.uk": ["Amazon.co.uk"],
    "Amazon.com.au": ["Amazon.com.au"],
    "Amazon.de": ["Amazon.de"],
    "Wholesale": pl.col("mode").eq("WHOLESALE"),
}


def filter_and_aggregate(
    start: Month,
    end: Month | None,
    input_df: pl.DataFrame,
    channel_filter: ChannelFilterKey,
    incomplete_years=[2024],
    ignore_years: list[int] = [],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    channels = CHANNEL_FILTERS[channel_filter]
    if not isinstance(channels, pl.Expr):
        parsed_channels: list[Channel] = [
            Channel.parse(x) for x in normalize_as_list(channels)
        ]
        input_df = struct_filter(
            input_df,
            *parsed_channels,
        )
    else:
        input_df = input_df.filter(channels)

    tpoints = pl.Series("date", [], dtype=pl.Date())
    period_month_points = range(start, 13 if end is None else end)
    valid_years = sorted(set([2022, 2023, 2024]).difference(ignore_years))

    for year in [x for x in valid_years]:
        if end is not None:
            tpoints.extend(
                TimePeriod(
                    Date.from_ymd(year, start, 1),
                    Date.from_ymd(year, end, 1),
                ).tpoints
            )
        else:
            tpoints.extend(
                TimePeriod(
                    Date.from_ymd(year, start, 1),
                    Date.from_ymd(year + 1, Month(1), 1),
                ).tpoints
            )
    period_df = input_df.filter(pl.col("date").is_in(tpoints))

    period_df = (
        period_df.with_columns(year=pl.col("date").dt.year())
        .drop("date")
        .select(
            ["channel", "sku", "category", "print", "size", "year", "sales"]
        )
    )

    period_month_points = tpoints.dt.month().unique()
    period_month_points = (
        period_month_points.filter(
            period_month_points.is_in(
                range(start, end if end is not None else calendar.DECEMBER + 1)
            )
        )
        .unique()
        .sort()
    )

    period_breakdown = (
        input_df.select(["category", "print", "size", "date", "sales"])
        .group_by(cs.exclude("sales"))
        .agg(pl.col("sales").sum())
        .with_columns(
            in_period=pl.col("date").dt.month().is_in(period_month_points),
            year=pl.col("date").dt.year(),
        )
        .drop("date")
        .filter(~pl.col("year").is_in(incomplete_years))
    )
    period_year_sales = (
        period_breakdown.drop("in_period")
        .group_by(cs.exclude("size", "sales"))
        .agg(annual_sales=pl.col("sales").sum())
    )
    in_period_sales = (
        period_breakdown.filter(pl.col("in_period"))
        .drop("in_period")
        .group_by(cs.exclude("size", "sales"))
        .agg(in_period_sales=pl.col("sales").sum())
    )
    period_breakdown = (
        period_year_sales.join(
            in_period_sales, on=["category", "print", "year"]
        )
        .with_columns(
            in_period_sales=pl.col("in_period_sales") / pl.col("annual_sales")
        )
        .drop("annual_sales")
    ).filter(pl.col("in_period_sales").is_not_nan())

    channel_breakdown = (
        period_df.select(["channel"] + ["category"] + ["year", "sales"])
        .group_by("channel", "year")
        .agg(pl.col("sales").sum(), pl.col("category").unique().first())
        .select("channel", "category", "year", "sales")
        .sort("sales", "channel", "year", descending=True)
    )

    sales_trend = (
        (
            period_df.drop("channel")
            .group_by(cs.exclude("sales"))
            .agg(pl.col("sales").sum())
            .with_columns(
                annual_sales=pl.col("sales")
                .sum()
                .over("category", "print", "year")
            )
            .with_columns(
                sales_proportion=(pl.col("sales") / pl.col("annual_sales"))
            )
        )
        .cast({"size": pl.Enum(["S", "M", "L", "XL"])})
        .sort("size", "year")
        .select(
            ["category", "print", "size"]
            + ["year"]
            + ["sales", "annual_sales", "sales_proportion"]
        )
    )
    return (sales_trend, channel_breakdown, period_breakdown)


@dataclass
class SalesTrend:
    period_sales_df: pl.DataFrame
    channel_breakdown_df: pl.DataFrame
    period_breakdown_df: pl.DataFrame
    channel_breakdown: alt.Chart
    absolute: alt.Chart
    normalized: alt.Chart
    proportion_period_sales: alt.Chart
    combined_title: str
    combined: alt.Chart | alt.ConcatChart | alt.HConcatChart

    def __init__(
        self,
        df: pl.DataFrame,
        start_month: Month,
        end_month: Month | None,
        channel_filter: ChannelFilterKey,
        ignore_years: list[int] = [],
    ):
        (
            self.period_sales_df,
            self.channel_breakdown_df,
            self.period_breakdown_df,
        ) = filter_and_aggregate(
            start_month,
            end_month,
            df,
            channel_filter,
            ignore_years=ignore_years,
        )

        if end_month is not None:
            period_title = f"from {calendar.month_abbr[start_month].upper()} to {calendar.month_abbr[end_month].upper()}"
        else:
            if start_month == calendar.JANUARY:
                period_title = "whole year"
            else:
                period_title = f"from {calendar.month_abbr[start_month].upper()} to next year JAN"

        print(period_title)
        print(channel_filter)
        self.channel_breakdown = self.channel_breakdown_df.plot.bar(
            x="year:N",
            y=alt.Y("sales:Q").title("total sales in period"),
            color=alt.Color("channel", scale=alt.Scale(scheme="category20")),
        ).properties(
            title=f"HCB0-LDG sales, {period_title}, for {channel_filter}"
        )

        # self.absolute = self.period_sales_df.plot.bar(
        #     x="size",
        #     y=Y("sales").title("qty"),
        #     xOffset="year:N",
        #     color=alt.Color(
        #         "year:N", scale=alt.Scale(scheme="yellowgreenblue")
        #     ),
        #     # order=alt.Order("size:O", sort="ascending"),
        # ).properties(title="absolute")

        # self.normalized = self.period_sales_df.plot.bar(
        #     x="size",
        #     y=Y("sales_proportion", scale=Scale(domain=[0.0, 1.0]))
        #     .title("sales proportion")
        #     .axis(format=".0%"),
        #     xOffset="year:N",
        #     color=alt.Color(
        #         "year:N", scale=alt.Scale(scheme="yellowgreenblue")
        #     ),
        #     # order=alt.Order("size:O", sort="ascending"),
        # ).properties(title="normalized (by total sales in period for year)")

        # self.combined_title = (
        #     f"HCB0-LDG sales, from {period_title}, for {channel_filter}"
        # )

        # self.proportion_period_sales = self.period_breakdown_df.plot.bar(
        #     x="year:O",
        #     y=Y("in_period_sales", scale=Scale(domain=[0.0, 1.0])).title(
        #         "proportion of yearly sales in period"
        #     ),
        # )
        # if end_month is not None:
        #     self.combined = (
        #         self.absolute | self.normalized | self.proportion_period_sales
        #     ).properties(title=self.combined_title)
        # else:
        #     self.combined = (self.absolute | self.normalized).properties(
        #         title=self.combined_title
        #     )
