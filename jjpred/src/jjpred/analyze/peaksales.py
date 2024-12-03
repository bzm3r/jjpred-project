import polars as pl

from jjpred.analysisdefn import AnalysisDefn
from jjpred.analyze.modelling.utils import (
    DuplicateEliminationStrategy,
    create_agg_label_default_dict,
    sum_quantity_in_order,
)
from jjpred.database import DataBase
from jjpred.sku import Sku
from jjpred.structlike import MemberType
from jjpred.utils.datetime import DateLike, Date, first_day
from jjpred.utils.typ import ScalarOrList, as_list


def calculate_peaks(
    agg_sales: pl.DataFrame,
    index_cols: ScalarOrList[str],
    num_digits_to_round_monthly_ratio: int | None,
    ignore_peak_at: DateLike | None,
) -> pl.DataFrame:
    if ignore_peak_at is not None:
        ignore_peak_at = Date.from_datelike(ignore_peak_at)
    else:
        ignore_peak_at = Date.from_datelike("0001-JAN-01")

    index_cols = as_list(index_cols)

    sales_with_peaks = (
        agg_sales.sort(index_cols + ["year", "month"])
        .with_columns(
            next_index=pl.struct(index_cols).shift(-1),
            prev_index=pl.struct(index_cols).shift(1),
        )
        .with_columns(
            diff_fwd=pl.when(pl.col.next_index.eq(pl.struct(index_cols)))
            .then(pl.col("sales").shift(-1) - pl.col("sales"))
            .otherwise(None),
        )
        .with_columns(
            is_peak=pl.when(
                pl.col.next_index.eq(pl.struct(index_cols))
                & pl.col.prev_index.eq(pl.struct(index_cols))
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
        .with_columns(
            peak_monthly_ratio=pl.when(pl.col.is_peak)
            .then(pl.col.monthly_ratio)
            .otherwise(pl.lit(0.0)),
            num_peaks=pl.col("is_peak").sum().over(pl.struct(index_cols)),
        )
    )

    peak_monthly_ratio_info = (
        sales_with_peaks.filter(pl.col.is_peak)
        .group_by(index_cols)
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
            index_cols
            + [
                "min_peak_monthly_ratio",
                "avg_peak_monthly_ratio",
                "max_peak_monthly_ratio",
            ]
        )
    )

    sales_with_peaks = (
        sales_with_peaks.join(
            peak_monthly_ratio_info, on=index_cols, how="left"
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
    index_cols: ScalarOrList[str],
    # filtered_isr_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    index_cols = as_list(index_cols)

    missing_months_filter = pl.col.year.eq(
        analysis_defn.date.year
    ) & pl.col.month.ge(analysis_defn.date.month)

    agg_sales = (
        filtered_history.select(
            index_cols + ["season", "year", "month", "sales"]
        )
        .group_by(index_cols + ["season", "year", "month"])
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
            annual_sales=pl.col("sales").sum().over(index_cols + ["year"]),
            max_month_sales=pl.col("sales").max().over(index_cols + ["year"]),
        )
        .sort("annual_sales", "month")
        .with_columns(
            is_max_month=(
                pl.col("sales").eq(pl.col("max_month_sales"))
                & pl.col.max_month_sales.gt(0)
            )
        )
    )

    agg_sales = agg_sales.with_columns(
        monthly_ratio=pl.col("sales") / pl.col("annual_sales"),
    )

    agg_sales = agg_sales.with_columns(
        date=pl.date(pl.col.year, pl.col.month, 1),
    )

    return agg_sales


def aggregate_relevant_history(
    analysis_defn: AnalysisDefn,
    relevant_history: pl.DataFrame,
    channel_filter_description: str,
) -> pl.DataFrame:
    agg_sales = sum_quantity_in_order(
        analysis_defn,
        relevant_history,
        DuplicateEliminationStrategy.MAX,
        "sales",
        ["channel", "print", "size"],
        create_agg_label_default_dict({"channel": channel_filter_description}),
    )
    # agg_sales = (
    #     relevant_history.select(
    #         ["channel"] + WHOLE_SKU_IDS + ["category", "date", "sales"]
    #     )
    #     .join(
    #         read_meta_info(analysis_defn, "all_sku").select(
    #             [c for c in ALL_SKU_IDS if c != "category"]
    #         ),
    #         on=WHOLE_SKU_IDS,
    #     )
    #     .select(
    #         ["channel"] + Sku.members(MemberType.SECONDARY) + ["date", "sales"]
    #     )
    # )

    # index_by_cols = ["channel", "print", "size", "sku_remainder"]

    # for index_col in reversed(index_by_cols):
    #     this_index_cols = ["category", "date"] + [
    #         ic for ic in index_by_cols if ic != index_col
    #     ]

    #     agg_sales = agg_sales.vstack(
    #         agg_sales.group_by(this_index_cols)
    #         .agg(pl.col.sales.sum())
    #         .with_columns(
    #             pl.lit("_ALL_", dtype=agg_sales[index_col].dtype).alias(
    #                 index_col
    #             )
    #         )
    #         .select(agg_sales.columns)
    #     )

    # missing_months_filter = pl.col.date.ge(analysis_defn.date.as_polars_date())
    # agg_sales.with_columns(
    #     sales=pl.when(missing_months_filter).then(None).otherwise(pl.col.sales)
    # )

    assert (
        len(
            agg_sales.filter(
                pl.col.sales.is_null()
                & pl.col.date.dt.year().ne(analysis_defn.date.year)
            )
        )
        == 0
    )

    primary_cols = ["channel"] + Sku.members(MemberType.PRIMARY)

    agg_sales = (
        agg_sales.with_columns(
            year=pl.col.date.dt.year(), month=pl.col.date.dt.month()
        )
        .with_columns(
            annual_sales=pl.col("sales").sum().over(primary_cols + ["year"]),
            max_month_sales=pl.col("sales")
            .max()
            .over(primary_cols + ["year"]),
        )
        .with_columns(
            is_max_month=(
                pl.col("sales").eq(pl.col("max_month_sales"))
                & pl.col.max_month_sales.gt(0)
            )
        )
        .with_columns(
            monthly_ratio=pl.col("sales") / pl.col("annual_sales"),
        )
    )

    return agg_sales


# def calculate_peaks_per_index(
#     analysis_defn: AnalysisDefn,
#     database: DataBase,
#     relevant_history: pl.DataFrame,
#     index_cols: ScalarOrList[str],
#     # isr_df: pl.DataFrame | None = None,
#     num_digits_to_round_monthly_ratio: int | None,
# ) -> pl.DataFrame:
#     season_info = database.meta_info.all_sku.select(
#         ["category"] + ["season"]
#     ).unique()

#     index_cols = as_list(index_cols)

#     # ["channel", "print", "size"]
#     for index in index_cols:
#         ...

#     channels = relevant_history["channel"].unique().sort()

#     agg_sales = aggregate_sales(
#         analysis_defn, relevant_history.drop("channel"), index_cols
#     ).with_columns(channel=pl.lit("_ALL_"))
#     sales_with_peaks = calculate_peaks(
#         agg_sales,
#         "category",
#         num_digits_to_round_monthly_ratio,
#         first_day(analysis_defn.date),
#     )

#     for channel in channels:
#         channel_agg_sales = aggregate_sales(
#             analysis_defn, relevant_history.filter(pl.col.channel.eq(channel))
#         ).with_columns(channel=pl.lit(channel))
#         channel_sales_with_peaks = calculate_peaks(
#             channel_agg_sales,
#             "category",
#             num_digits_to_round_monthly_ratio,
#             first_day(analysis_defn.date),
#         )
#         sales_with_peaks = sales_with_peaks.vstack(channel_sales_with_peaks)

#     return sales_with_peaks.join(
#         season_info,
#         on="category",
#         validate="m:1",
#     )


def calculate_peaks_per_channel(
    analysis_defn: AnalysisDefn,
    db: DataBase,
    relevant_history: pl.DataFrame,
    channel_filter_description: str,
    num_digits_to_round_monthly_ratio: int | None,
) -> pl.DataFrame:
    agg_sales = aggregate_relevant_history(
        analysis_defn, relevant_history, channel_filter_description
    )

    sales_with_peaks = calculate_peaks(
        agg_sales,
        ["channel", "category", "print", "size"],
        num_digits_to_round_monthly_ratio,
        first_day(analysis_defn.date),
    )

    season_info = db.meta_info.all_sku.select(
        ["category"] + ["season"]
    ).unique()

    return sales_with_peaks.join(
        season_info,
        on="category",
        validate="m:1",
    )


def calculate_peaks_per_channel_old(
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
        analysis_defn, relevant_history.drop("channel"), ["category"]
    ).with_columns(channel=pl.lit("_ALL_"))

    sales_with_peaks = calculate_peaks(
        agg_sales,
        "category",
        num_digits_to_round_monthly_ratio,
        first_day(analysis_defn.date),
    )

    for channel in channels:
        channel_agg_sales = aggregate_sales(
            analysis_defn,
            relevant_history.filter(pl.col.channel.eq(channel)),
            ["category"],
        ).with_columns(channel=pl.lit(channel))
        channel_sales_with_peaks = calculate_peaks(
            channel_agg_sales,
            "category",
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
#             next_index=pl.col("category").shift(-1),
#             prev_category=pl.col("category").shift(1),
#         )
#         .with_columns(
#             diff_fwd=pl.when(pl.col.next_index.eq(pl.col.category))
#             .then(pl.col("fake_sales").shift(-1) - pl.col("fake_sales"))
#             .otherwise(None),
#         )
#         .with_columns(
#             is_peak=pl.when(
#                 pl.col.next_index.eq(pl.col.category)
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
#             annual_sales=pl.col("sales").sum().over("category", "year"),
#             fake_year_sales=pl.col("fake_sales")
#             .sum()
#             .over("category", "year"),
#             max_month_sales=pl.col("sales").max().over("category", "year"),
#         )
#         .sort("annual_sales", "month")
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
#     ).with_columns(channel=pl.lit("_ALL_"))
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
