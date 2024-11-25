import polars as pl

from jjpred.analysisdefn import AnalysisDefn
from jjpred.analyze.modelling.utils import (
    ChannelFilter,
    DuplicateEliminationStrategy,
    create_agg_label_default_dict,
    sum_quantity_in_order,
    get_analysis_defn_and_db,
)
from jjpred.channel import Channel
from jjpred.database import DataBase
from jjpred.readsheet import DataVariant


def get_isr_info(
    analysis_defn_or_db: AnalysisDefn | DataBase,
    channel_filter: ChannelFilter,
    read_db_from_disk: bool = True,
) -> pl.DataFrame:
    # raw_isr = read_in_stock_ratios(db, read_from_disk=False)
    _, db = get_analysis_defn_and_db(
        analysis_defn_or_db, read_db_from_disk=read_db_from_disk
    )

    isr_df = (
        db.dfs[DataVariant.InStockRatio]
        .filter(channel_filter.expression)
        .drop(Channel.members())
        .with_columns(
            in_stock_ratio=pl.when(
                pl.col.date.lt(
                    pl.date(
                        db.analysis_defn.date.year,
                        db.analysis_defn.date.month,
                        1,
                    )
                )
            )
            .then(pl.col.in_stock_ratio)
            .otherwise(pl.lit(None))
        )
    )

    # local_warehouse_dependent = [
    #     Channel.parse(x).pretty_string_repr()
    #     for x in ["janandjul.com", "Wholesale", "Vancouver Showroom"]
    # ]

    # local_warehouse_isr = isr_df.filter(
    #     pl.col.channel.eq(Channel.parse("warehouse").pretty_string_repr())
    # ).drop("channel")

    # isr_df_channels = isr_df["channel"].unique()

    # for channel in local_warehouse_dependent:
    #     if channel not in isr_df_channels:
    #         isr_df = isr_df.vstack(
    #             local_warehouse_isr.with_columns(
    #                 channel=pl.lit(channel, dtype=isr_df["channel"].dtype)
    #             ).select(isr_df.columns)
    #         )

    isr_df_with_year = isr_df.with_columns(
        year=pl.col.date.dt.year()
    ).with_columns(
        max_year_isr=pl.col.in_stock_ratio.max().over(
            "a_sku", "sku", "category", "channel", "year"
        ),
    )

    return isr_df_with_year


def get_aggregating_expr_for_agg_by_col(agg_by_col: str) -> pl.Expr:
    if agg_by_col == "sku_remainder":
        return pl.col.in_stock_ratio.max()
    else:
        return pl.col.in_stock_ratio.mean()


def get_mean_non_zero_isr(
    analysis_defn_or_db: AnalysisDefn | DataBase, channel_filter: ChannelFilter
) -> pl.DataFrame:
    analysis_defn, db = get_analysis_defn_and_db(analysis_defn_or_db)
    agg_isr = get_isr_info(db, channel_filter)

    agg_isr = agg_isr.filter(pl.col.max_year_isr.gt(0.0)).with_columns(
        count=pl.lit(1)
    )

    jj_pretty_string = Channel.parse("janandjul.com").pretty_string_repr()
    if jj_pretty_string in agg_isr["channel"].unique():
        # If "janandjul.com" is in the channels of this dataframe, then its
        # in-stock ratio values (same as local warehouse) should be taken as the
        # aggregated in-stock ratio
        agg_isr = agg_isr.vstack(
            agg_isr.filter(pl.col.channel.eq(jj_pretty_string)).with_columns(
                channel=pl.lit(
                    channel_filter.description, dtype=agg_isr["channel"].dtype
                )
            )
        )

        agg_isr = (
            sum_quantity_in_order(
                analysis_defn,
                agg_isr,
                DuplicateEliminationStrategy.MAX,
                ["in_stock_ratio", "count"],
                ["print", "size", "sku_remainder"],
                create_agg_label_default_dict(
                    {"channel": channel_filter.description}
                ),
            )
            .with_columns(in_stock_ratio=pl.col.in_stock_ratio / pl.col.count)
            .drop("count")
        )
    else:
        agg_isr = (
            sum_quantity_in_order(
                analysis_defn,
                agg_isr,
                DuplicateEliminationStrategy.MAX,
                ["in_stock_ratio", "count"],
                ["channel", "print", "size", "sku_remainder"],
                create_agg_label_default_dict(
                    {"channel": channel_filter.description}
                ),
            )
            .with_columns(in_stock_ratio=pl.col.in_stock_ratio / pl.col.count)
            .drop("count")
        )

    # agg_isr = (
    #     agg_isr.select(
    #         ["channel"]
    #         + WHOLE_SKU_IDS
    #         + ["category", "date", "in_stock_ratio"]
    #     )
    #     .join(
    #         read_meta_info(analysis_defn, "all_sku").select(
    #             [c for c in ALL_SKU_IDS if c != "category"]
    #         ),
    #         on=WHOLE_SKU_IDS,
    #     )
    #     .select(
    #         ["channel"]
    #         + Sku.members(MemberType.SECONDARY)
    #         + ["date", "in_stock_ratio"]
    #     )
    # )

    # index_by_cols = ["print", "size", "sku_remainder"]

    # for index_col in reversed(index_by_cols):
    #     this_index_cols = ["category", "date"] + [
    #         ic for ic in index_by_cols if ic != index_col
    #     ]

    #     if index_col == "sku_remainder":
    #         aggregator_expr = pl.col.in_stock_ratio.max()
    #     else:
    #         aggregator_expr = pl.col.in_stock_ratio.mean()

    #     agg_isr = agg_isr.vstack(
    #         agg_isr.group_by(this_index_cols)
    #         .agg(aggregator_expr)
    #         .with_columns(
    #             pl.lit("_ALL_", dtype=agg_isr[index_col].dtype).alias(
    #                 index_col
    #             )
    #         )
    #         .select(agg_isr.columns)
    #     )

    # channel_dtype = pl.Enum(agg_isr["channel"].unique().sort())

    # mean_non_zero_isr_per_cat = agg_isr.cast({"channel": channel_dtype})

    return agg_isr


# def get_mean_non_zero_isr_per_cat(isr_info: pl.DataFrame) -> pl.DataFrame:
#     mean_non_zero_isr_per_cat = (
#         isr_info.filter(pl.col.max_year_isr.gt(0.0))
#         .group_by("category", "channel", "date")
#         .agg(pl.col.in_stock_ratio.mean())
#     )

#     mean_non_zero_isr_per_cat = mean_non_zero_isr_per_cat.vstack(
#         mean_non_zero_isr_per_cat.filter(
#             pl.col.channel.eq("janandjul.com")
#         ).with_columns(channel=pl.lit("_ALL_"))
#     )

#     channel_dtype = pl.Enum(
#         mean_non_zero_isr_per_cat["channel"].unique().sort()
#     )

#     mean_non_zero_isr_per_cat = mean_non_zero_isr_per_cat.cast(
#         {"channel": channel_dtype}
#     )

#     return mean_non_zero_isr_per_cat
