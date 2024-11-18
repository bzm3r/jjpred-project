import polars as pl

from jjpred.analysisdefn import AnalysisDefn
from jjpred.database import DataBase
from jjpred.utils.datetime import DateLike, Date, first_day


def calculate_peaks(
    agg_sales: pl.DataFrame,
    num_digits_to_round_monthly_ratio: int | None,
    ignore_peak_at: DateLike | None,
) -> pl.DataFrame:
    if ignore_peak_at is not None:
        ignore_peak_at = Date.from_datelike(ignore_peak_at)
    else:
        ignore_peak_at = Date.from_datelike("0001-JAN-01")

    sales_with_peaks = (
        agg_sales.sort("category", "year", "month")
        .with_columns(
            next_category=pl.col("category").shift(-1),
            prev_category=pl.col("category").shift(1),
        )
        .with_columns(
            diff_fwd=pl.when(pl.col.next_category.eq(pl.col.category))
            .then(pl.col("sales").shift(-1) - pl.col("sales"))
            .otherwise(None),
        )
        .with_columns(
            is_peak=pl.when(
                pl.col.next_category.eq(pl.col.category)
                & pl.col.prev_category.eq(pl.col.category)
                & (
                    pl.col.sales.is_not_null()
                    & pl.col.sales.shift(-1).is_not_null()
                )
            )
            .then(pl.col("diff_fwd").lt(0) & pl.col("diff_fwd").shift(1).gt(0))
            .otherwise(
                pl.col("is_max_month")
                & ~(
                    pl.col.year.eq(ignore_peak_at.year)
                    & pl.col.month.eq(ignore_peak_at.month)
                )
            )
        )
        .with_columns(num_peaks=pl.col("is_peak").sum().over("category"))
    )

    peak_monthly_ratio_info = (
        sales_with_peaks.filter(pl.col.is_peak)
        .group_by("category")
        .agg(
            pl.col.monthly_ratio.sum().alias("total_peak_monthly_ratio"),
            pl.col.monthly_ratio.min().alias("min_peak_monthly_ratio"),
            pl.col.monthly_ratio.max().alias("max_peak_monthly_ratio"),
            pl.col.num_peaks.max(),
        )
        .with_columns(
            avg_peak_monthly_ratio=pl.col.total_peak_monthly_ratio
            / pl.col.num_peaks
        )
        .select(
            "category",
            "min_peak_monthly_ratio",
            "avg_peak_monthly_ratio",
            "max_peak_monthly_ratio",
        )
    )

    sales_with_peaks = (
        sales_with_peaks.join(
            peak_monthly_ratio_info, on=["category"], how="left"
        )
        .with_columns(
            pl.col.monthly_ratio.round(3),
        )
        .with_columns(
            peak_label=pl.when(pl.col.is_peak)
            .then(pl.lit("peak"))
            .otherwise(pl.lit("")),
        )
    )

    if num_digits_to_round_monthly_ratio is not None:
        assert num_digits_to_round_monthly_ratio > 0

        sales_with_peaks = sales_with_peaks.with_columns(
            pl.col.monthly_ratio.round(num_digits_to_round_monthly_ratio)
        )

    sales_with_peaks = sales_with_peaks.with_columns(
        peak_monthly_ratio=pl.when(pl.col.is_peak)
        .then(pl.col.monthly_ratio)
        .otherwise(pl.lit(0.0)),
    )

    return sales_with_peaks


def aggregate_sales(
    analysis_defn: AnalysisDefn,
    filtered_history: pl.DataFrame,
    # filtered_isr_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    missing_months_filter = pl.col.year.eq(
        analysis_defn.date.year
    ) & pl.col.month.ge(analysis_defn.date.month)
    agg_sales = (
        filtered_history.select(
            ["category", "season"] + ["year", "month", "sales"]
        )
        .group_by("category", "season", "year", "month")
        .agg(pl.col("sales").sum())
        .with_columns(
            sales=pl.when(missing_months_filter)
            .then(None)
            .otherwise(pl.col.sales)
        )
    )
    assert (
        len(
            agg_sales.filter(
                pl.col.sales.is_null()
                & pl.col.year.ne(analysis_defn.date.year)
            )
        )
        == 0
    )

    agg_sales = (
        agg_sales.with_columns(
            year_sales=pl.col("sales").sum().over("category", "year"),
            max_month_sales=pl.col("sales").max().over("category", "year"),
        )
        .sort("year_sales", "month")
        .with_columns(
            is_max_month=(
                pl.col("sales").eq(pl.col("max_month_sales"))
                & pl.col.max_month_sales.gt(0)
            )
        )
    )

    agg_sales = agg_sales.with_columns(
        monthly_ratio=pl.col("sales") / pl.col("year_sales"),
    )

    agg_sales = agg_sales.with_columns(
        date=pl.date(pl.col.year, pl.col.month, 1),
    )

    return agg_sales


def calculate_peaks_per_channel(
    analysis_defn: AnalysisDefn,
    database: DataBase,
    relevant_history: pl.DataFrame,
    # isr_df: pl.DataFrame | None = None,
    num_digits_to_round_monthly_ratio: int | None,
) -> pl.DataFrame:
    season_info = database.meta_info.all_sku.select(
        ["category"] + ["season"]
    ).unique()

    channels = relevant_history["channel"].unique().sort()

    agg_sales = aggregate_sales(
        analysis_defn, relevant_history.drop("channel")
    ).with_columns(channel=pl.lit("ALL"))
    sales_with_peaks = calculate_peaks(
        agg_sales,
        num_digits_to_round_monthly_ratio,
        first_day(analysis_defn.date),
    )

    for channel in channels:
        channel_agg_sales = aggregate_sales(
            analysis_defn, relevant_history.filter(pl.col.channel.eq(channel))
        ).with_columns(channel=pl.lit(channel))
        channel_sales_with_peaks = calculate_peaks(
            channel_agg_sales,
            num_digits_to_round_monthly_ratio,
            first_day(analysis_defn.date),
        )
        sales_with_peaks = sales_with_peaks.vstack(channel_sales_with_peaks)

    return sales_with_peaks.join(
        season_info,
        on="category",
        validate="m:1",
    )


# def calculate_peaks(
#     agg_sales: pl.DataFrame,
#     num_digits_to_round_monthly_ratio: int | None,
#     ignore_peak_at: DateLike | None,
# ) -> pl.DataFrame:
#     if ignore_peak_at is not None:
#         ignore_peak_at = Date.from_datelike(ignore_peak_at)
#     else:
#         ignore_peak_at = Date.from_datelike("0001-JAN-01")

#     sales_with_peaks = (
#         agg_sales.sort("category", "year", "month")
#         .with_columns(
#             next_category=pl.col("category").shift(-1),
#             prev_category=pl.col("category").shift(1),
#         )
#         .with_columns(
#             diff_fwd=pl.when(pl.col.next_category.eq(pl.col.category))
#             .then(pl.col("fake_sales").shift(-1) - pl.col("fake_sales"))
#             .otherwise(None),
#         )
#         .with_columns(
#             is_peak=pl.when(
#                 pl.col.next_category.eq(pl.col.category)
#                 & pl.col.prev_category.eq(pl.col.category)
#                 & (
#                     pl.col.sales.is_not_null()
#                     & pl.col.sales.shift(-1).is_not_null()
#                 )
#             )
#             .then(pl.col("diff_fwd").lt(0) & pl.col("diff_fwd").shift(1).gt(0))
#             .otherwise(
#                 pl.col("is_max_month")
#                 & ~(
#                     pl.col.year.eq(ignore_peak_at.year)
#                     & pl.col.month.eq(ignore_peak_at.month)
#                 )
#             )
#         )
#         .with_columns(num_peaks=pl.col("is_peak").sum().over("category"))
#     )

#     peak_monthly_ratio_info = (
#         sales_with_peaks.filter(pl.col.is_peak)
#         .group_by("category")
#         .agg(
#             pl.col.monthly_ratio.sum().alias("total_peak_monthly_ratio"),
#             pl.col.monthly_ratio.min().alias("min_peak_monthly_ratio"),
#             pl.col.monthly_ratio.max().alias("max_peak_monthly_ratio"),
#             pl.col.num_peaks.max(),
#         )
#         .with_columns(
#             avg_peak_monthly_ratio=pl.col.total_peak_monthly_ratio
#             / pl.col.num_peaks
#         )
#         .select(
#             "category",
#             "min_peak_monthly_ratio",
#             "avg_peak_monthly_ratio",
#             "max_peak_monthly_ratio",
#         )
#     )

#     sales_with_peaks = (
#         sales_with_peaks.join(
#             peak_monthly_ratio_info, on=["category"], how="left"
#         )
#         .with_columns(
#             pl.col.monthly_ratio.round(3),
#         )
#         .with_columns(
#             peak_label=pl.when(pl.col.is_peak)
#             .then(pl.lit("peak"))
#             .otherwise(pl.lit("")),
#         )
#     )

#     if num_digits_to_round_monthly_ratio is not None:
#         assert num_digits_to_round_monthly_ratio > 0

#         sales_with_peaks = sales_with_peaks.with_columns(
#             pl.col.monthly_ratio.round(num_digits_to_round_monthly_ratio)
#         )

#     sales_with_peaks = sales_with_peaks.with_columns(
#         peak_monthly_ratio=pl.when(pl.col.is_peak)
#         .then(pl.col.monthly_ratio)
#         .otherwise(pl.lit(0.0)),
#     )

#     return sales_with_peaks


# def aggregate_sales(
#     analysis_defn: AnalysisDefn,
#     filtered_history: pl.DataFrame,
#     # filtered_isr_df: pl.DataFrame | None = None,
# ) -> pl.DataFrame:
#     missing_months_filter = pl.col.year.eq(
#         analysis_defn.date.year
#     ) & pl.col.month.ge(analysis_defn.date.month)
#     agg_sales = (
#         filtered_history.select(
#             ["category", "season"] + ["year", "month", "sales"]
#         )
#         .group_by("category", "season", "year", "month")
#         .agg(pl.col("sales").sum())
#         .with_columns(
#             sales=pl.when(missing_months_filter)
#             .then(None)
#             .otherwise(pl.col.sales)
#         )
#     )
#     assert (
#         len(
#             agg_sales.filter(
#                 pl.col.sales.is_null()
#                 & pl.col.year.ne(analysis_defn.date.year)
#             )
#         )
#         == 0
#     )

#     last_year_missing_month_sales = (
#         agg_sales.filter(pl.col.year.eq(analysis_defn.date.year - 1))
#         .select("category", "month", "year", "sales")
#         .rename({"sales": "alt_sales"})
#         .with_columns(
#             alt_year_sales=pl.col.alt_sales.sum().over("category", "year")
#         )
#         .with_columns(
#             alt_sales=pl.when(pl.col.alt_year_sales.gt(0))
#             .then(pl.col.alt_sales)
#             .otherwise(None)
#         )
#     )
#     agg_sales = (
#         agg_sales.join(
#             last_year_missing_month_sales.select(
#                 "category", "month", "alt_sales", "alt_year_sales"
#             ),
#             on=["category", "month"],
#             how="left",
#         )
#         .with_columns(
#             fake_sales=pl.when(
#                 pl.col.sales.is_null() & pl.col.alt_year_sales.gt(0)
#             )
#             .then(pl.col.alt_sales)
#             .otherwise(pl.col.sales)
#         )
#         .with_columns(
#             year_sales=pl.col("sales").sum().over("category", "year"),
#             fake_year_sales=pl.col("fake_sales")
#             .sum()
#             .over("category", "year"),
#             max_month_sales=pl.col("sales").max().over("category", "year"),
#         )
#         .sort("year_sales", "month")
#         .with_columns(
#             is_max_month=(
#                 pl.col("sales").eq(pl.col("max_month_sales"))
#                 & pl.col.max_month_sales.gt(0)
#             )
#         )
#     )

#     agg_sales = agg_sales.with_columns(
#         monthly_ratio=pl.col("fake_sales") / pl.col("fake_year_sales"),
#     )

#     agg_sales = agg_sales.with_columns(
#         date=pl.date(pl.col.year, pl.col.month, 1),
#     )

#     return agg_sales


# def calculate_peaks_per_channel(
#     analysis_defn: AnalysisDefn,
#     database: DataBase,
#     relevant_history: pl.DataFrame,
#     # isr_df: pl.DataFrame | None = None,
#     num_digits_to_round_monthly_ratio: int | None,
# ) -> pl.DataFrame:
#     season_info = database.meta_info.all_sku.select(
#         ["category"] + ["season"]
#     ).unique()

#     channels = relevant_history["channel"].unique().sort()

#     agg_sales = aggregate_sales(
#         analysis_defn, relevant_history.drop("channel")
#     ).with_columns(channel=pl.lit("ALL"))
#     sales_with_peaks = calculate_peaks(
#         agg_sales,
#         num_digits_to_round_monthly_ratio,
#         first_day(analysis_defn.date),
#     )

#     for channel in channels:
#         channel_agg_sales = aggregate_sales(
#             analysis_defn, relevant_history.filter(pl.col.channel.eq(channel))
#         ).with_columns(channel=pl.lit(channel))
#         channel_sales_with_peaks = calculate_peaks(
#             channel_agg_sales,
#             num_digits_to_round_monthly_ratio,
#             first_day(analysis_defn.date),
#         )
#         sales_with_peaks = sales_with_peaks.vstack(channel_sales_with_peaks)

#     return sales_with_peaks.join(
#         season_info,
#         on="category",
#         validate="m:1",
#     )
