"""Functions to read in-stock ratios from the
``All Marketplace by MSKU - InStockRatio`` file."""

from __future__ import annotations
from pathlib import Path

import polars as pl
import polars.selectors as cs
import fastexcel as fxl

from jjpred.analysisdefn import AnalysisDefn
from jjpred.channel import Channel
from jjpred.datagroups import WHOLE_SKU_IDS
from jjpred.globalpaths import ANALYSIS_INPUT_FOLDER
from jjpred.readsheet import DataVariant, get_relevant_sheets
from jjpred.readsupport.mastersku import MasterSkuInfo
from jjpred.readsupport.utils import (
    cast_standard,
    parse_channels,
    unpivot_dates,
)
from jjpred.structlike import MemberType
from jjpred.utils.fileio import (
    delete_or_read_df,
    gen_isr_year_info_path,
    gen_support_info_path,
    read_df,
    write_df,
)
from jjpred.utils.polars import (
    binary_partition_strict,
    concat_enum_extend_vstack_strict,
    find_dupes,
)
from jjpred.utils.datetime import Date, DateLike


ISR_EXCEL_PATH_TEMPLATE = (
    "All Marketplace by MSKU - InStockRatio - {date}.xlsx"
)
"""In-Stock Ratio file should have format like:

``All Marketplace by MSKU - InStockRatio - {date}.xlsx``

Where the date has format: ``YYYY-MMM-DD``. For example: ``2024-OCT-21``.
"""

COMPLETE_ISR_PATH_TEMPLATE = "complete_isr_{date}.parquet"
"""Complete In Stock Ratio parquet file should have format like:

``complete_isr_{date}.paruqet``

Where the date has format: ``YYYY-MMM-DD``. For example: ``2024-OCT-21``.
"""


def gen_isr_excel_path(date: DateLike) -> Path:
    file_path = Path(
        ISR_EXCEL_PATH_TEMPLATE.format(
            date=Date.from_datelike(date).fmt_default()
        )
    )
    return ANALYSIS_INPUT_FOLDER.joinpath(file_path)


def gen_complete_isr_path(date: DateLike) -> Path:
    file_path = Path(
        COMPLETE_ISR_PATH_TEMPLATE.format(
            date=Date.from_datelike(date).fmt_default()
        )
    )
    return ANALYSIS_INPUT_FOLDER.joinpath(file_path)


def read_isr_year_info_raw(
    raw_isr_date: DateLike, years: int | list[int] | None
):
    """Read in-stock ratios from the ``All Marketplace by MSKU - InStockRatio``
    file, given required meta-information directly."""

    raise NotImplementedError(
        "Requires changes due to changes in ISR Excel file format "
        "(particularly: what kind of channels there are. there is "
        "now WH Richmond and Unknown Warehouse)"
    )

    excel_path = gen_isr_excel_path(raw_isr_date)

    unified_sheet = None

    wb = fxl.read_excel(excel_path)
    relevant_sheets = get_relevant_sheets(wb, DataVariant.InStockRatio, [])

    sheets = list(
        DataVariant.InStockRatio.extract_data(
            wb, relevant_sheets[DataVariant.InStockRatio]
        ).values()
    )
    assert len(sheets) > 0
    if len(sheets) > 0:
        unified_sheet = sheets[0]
        for other in sheets[1:]:
            unified_sheet.df = pl.concat(
                [unified_sheet.df, other.df], how="vertical"
            )
    assert unified_sheet is not None

    local_warehouse_df, non_warehouse_df = binary_partition_strict(
        unified_sheet.df, pl.col.channel.eq("JJ Warehouse")
    )
    local_warehouse_df = local_warehouse_df.drop("channel").with_columns(
        channel=pl.lit("Warehouse CA")
    )
    # jj_df = local_warehouse_df.drop("channel").with_columns(
    #     channel=pl.lit("janandjul.com")
    # )

    local_warehouse_dependent = [
        "janandjul.com",
        "Wholesale",
        "Faire.com",
        "Vancouver Showroom",
    ]
    extra_dfs = []
    for channel in local_warehouse_dependent:
        extra_dfs.append(
            local_warehouse_df.with_columns(
                channel=pl.lit(
                    channel, dtype=local_warehouse_df["channel"].dtype
                )
            ).select(non_warehouse_df.columns)
        )

    unified_sheet.df = pl.concat(
        [non_warehouse_df, local_warehouse_df.select(non_warehouse_df.columns)]
        + extra_dfs
    )
    # unified_sheet.df = other_df.vstack(
    #     warehouse_df.select(other_df.columns)
    # ).vstack(jj_df.select(other_df.columns))

    isr_df = unpivot_dates(
        unified_sheet.df,
        unified_sheet.id_cols,
        unified_sheet.data_cols,
        "in_stock_days",
    )
    del unified_sheet

    # remove multiple entries with all the same identifying
    # information yet different information
    isr_df = isr_df.group_by(cs.exclude("in_stock_days")).agg(
        pl.col("in_stock_days").sum()
    )

    if years is not None:
        if not isinstance(years, list):
            years = [years]

        isr_df = isr_df.filter(pl.col.date.dt.year().is_in(years))

    return isr_df


def process_isr_year_info_raw(
    master_sku_info: MasterSkuInfo,
    raw_isr: pl.DataFrame,
) -> pl.DataFrame:
    # remove any data for SKUs which have 0 in-stock-days recorded
    # over all channels and dates, or which are otherwise to be ignored, and
    # therefore not in all_sku_info
    isr_df = (
        raw_isr.with_columns(
            agg_sum=pl.col("in_stock_days")
            .sum()
            .over(
                cs.expand_selector(
                    raw_isr,
                    cs.exclude("in_stock_days", "channel", "date"),
                )
            ),
            is_active=pl.col("a_sku").is_in(
                master_sku_info.all_sku["a_sku"].unique()
            ),
        )
        .filter(pl.col("agg_sum").gt(0).or_(pl.col("is_active")))
        .drop("agg_sum", "is_active")
    )

    date_info = (
        isr_df.select("date")
        .unique()
        .with_columns(pl.col("date").dt.month_end().alias("month_end_date"))
        .with_columns(
            (pl.col("month_end_date") - pl.col("date") + pl.duration(days=1))
            .alias("days_in_month")
            .dt.total_days()
            .cast(pl.Int16())
        )
        .drop("month_end_date")
    )

    isr_df = isr_df.join(date_info, on="date").with_columns(
        in_stock_ratio=pl.col.in_stock_days / pl.col.days_in_month
    )

    isr_df = parse_channels(isr_df)

    isr_df = cast_standard(
        [master_sku_info.active_sku],
        isr_df.rename({"m_sku": "sku"}),
        strict=False,  # skip any SKUs that are not in the active_sku_info list
    ).filter(~(pl.col.sku.is_null() | pl.col.a_sku.is_null()))

    isr_df = isr_df.join(
        master_sku_info.active_sku.select(WHOLE_SKU_IDS + ["a_category"]),
        on=["sku", "a_sku"],
    ).rename({"a_category": "category"})

    # channel_info = channel_info.filter(
    #     pl.col.channel.is_in(isr_df["channel"].unique())
    # )

    # cast_standard([channel_info], isr_df)
    isr_df = isr_df.select(
        WHOLE_SKU_IDS
        + ["category"]
        + Channel.members(MemberType.META)
        + ["date", "in_stock_days", "days_in_month", "in_stock_ratio"]
    )

    return isr_df


def read_isr_year_info(
    analysis_defn: AnalysisDefn,
    raw_isr_date: DateLike,
    read_master_sku_info_from_disk: bool = True,
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
    years: int | list[int] | None = None,
    overwrite: bool = True,
) -> pl.DataFrame:
    raise NotImplementedError(
        "Requires changes due to changes in ISR Excel file format "
        "(particularly: what kind of channels there are. there is "
        "now WH Richmond and Unknown Warehouse)"
    )
    isr_path = gen_isr_year_info_path(years)

    if read_from_disk or delete_if_exists:
        isr_df = delete_or_read_df(delete_if_exists, isr_path)
        # per_cat = delete_or_read_df(delete_if_exists, per_cat_path)

        if isr_df is not None:  # and per_cat is not None:
            return isr_df

    raw_isr = read_isr_year_info_raw(raw_isr_date, years)

    master_sku_info = MasterSkuInfo.get_master_sku_info(
        analysis_defn,
        read_from_disk=read_master_sku_info_from_disk,
        overwrite=overwrite,
    )

    isr_df = process_isr_year_info_raw(master_sku_info, raw_isr)

    write_df(overwrite, isr_path, isr_df)

    return isr_df


def read_complete_isr_info(analysis_defn: AnalysisDefn) -> pl.DataFrame:
    analaysis_isr_date = analysis_defn.in_stock_ratio_date
    assert analaysis_isr_date is not None

    complete_isr_path = gen_complete_isr_path(analaysis_isr_date)

    complete_isr = read_df(complete_isr_path)

    return complete_isr


def read_isr_from_excel_file_given_meta_info(
    analysis_defn: AnalysisDefn,
    active_sku_info: pl.DataFrame,
    all_sku_info: pl.DataFrame,
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
    overwrite: bool = True,
) -> pl.DataFrame:
    """Read in-stock ratios from the ``All Marketplace by MSKU - InStockRatio``
    file, given required meta-information directly."""

    assert analysis_defn.in_stock_ratio_date is not None

    isr_path = gen_support_info_path(
        analysis_defn,
        "in_stock_ratio",
        analysis_defn.in_stock_ratio_date,
        source_name="isrfile",
    )

    if read_from_disk or delete_if_exists:
        isr_df = delete_or_read_df(delete_if_exists, isr_path)

        if isr_df is not None:
            return isr_df

    excel_path = gen_isr_excel_path(analysis_defn.in_stock_ratio_date)

    print(f"Reading ISR from: {excel_path}")

    unified_sheet = None

    wb = fxl.read_excel(excel_path)
    relevant_sheets = get_relevant_sheets(wb, DataVariant.InStockRatio, [])

    sheets = list(
        DataVariant.InStockRatio.extract_data(
            wb, relevant_sheets[DataVariant.InStockRatio]
        ).values()
    )
    assert len(sheets) > 0
    if len(sheets) > 0:
        unified_sheet = sheets[0]
        for other in sheets[1:]:
            unified_sheet.df = pl.concat(
                [unified_sheet.df, other.df], how="vertical"
            )
    assert unified_sheet is not None

    unknown_warehouse_channel = Channel.parse("unknown warehouse")
    local_warehouse_channel = Channel.parse("wh surrey")

    unified_sheet.df = (
        parse_channels(unified_sheet.df)
        .filter(
            ~pl.struct(Channel.members()).eq(
                unknown_warehouse_channel.as_dict()
            )
        )
        .drop("raw_channel")
    )

    find_dupes(
        unified_sheet.df,
        ["m_sku", "a_sku"] + Channel.members(MemberType.META),
        raise_error=True,
    )

    local_warehouse_df, non_warehouse_df = binary_partition_strict(
        unified_sheet.df,
        pl.struct(Channel.members()).eq(local_warehouse_channel.as_dict()),
    )
    local_warehouse_dependent_df = local_warehouse_df.drop(*Channel.members())

    local_warehouse_dependent = [
        "janandjul.com",
        "Wholesale",
        "Faire.com",
        "Vancouver Showroom",
    ]
    extra_dfs = []
    for channel in local_warehouse_dependent:
        extra_dfs.append(
            parse_channels(
                local_warehouse_dependent_df.with_columns(
                    channel=pl.lit(channel)
                )
            )
            .drop("raw_channel")
            .select(non_warehouse_df.columns)
        )

    unified_sheet.df = concat_enum_extend_vstack_strict(
        [non_warehouse_df, local_warehouse_df.select(non_warehouse_df.columns)]
        + extra_dfs
    )

    isr_df = unpivot_dates(
        unified_sheet.df,
        ["m_sku", "a_sku"] + Channel.members(MemberType.META),
        unified_sheet.data_cols,
        "in_stock_days",
    )
    del unified_sheet

    # remove multiple entries with all the same identifying
    # information yet different information
    find_dupes(
        isr_df,
        cs.expand_selector(isr_df, cs.exclude("in_stock_days")),
        raise_error=True,
    )

    # remove any data for SKUs which have 0 in-stock-days recorded
    # over all channels and dates, or which are otherwise to be ignored, and
    # therefore not in all_sku_info
    isr_df = (
        isr_df.with_columns(
            agg_sum=pl.col("in_stock_days")
            .sum()
            .over(
                cs.expand_selector(
                    isr_df,
                    cs.exclude(
                        "in_stock_days",
                        "date",
                        *Channel.members(MemberType.META),
                    ),
                )
            ),
            is_active=pl.col("a_sku").is_in(
                all_sku_info["a_sku"].unique().cast(pl.String())
            ),
        )
        .filter(pl.col("agg_sum").gt(0).or_(pl.col("is_active")))
        .drop("agg_sum", "is_active")
    )

    date_info = (
        isr_df.select("date")
        .with_columns(
            data_date=Date.from_datelike(
                pl.DataFrame()
                .with_columns(
                    date=analysis_defn.in_stock_ratio_date.as_polars_date()
                )
                .with_columns(day=pl.col.date.dt.weekday())
                .with_columns(
                    shifted_date=pl.col.date
                    + (-1 * pl.duration(days=pl.col.day - 1))
                )
                .with_columns(shifted_day=pl.col.shifted_date.dt.weekday())[
                    "shifted_date"
                ]
                .item()
            ).as_polars_date()
        )
        .unique()
        .with_columns(pl.col("date").dt.month_end().alias("month_end_date"))
        .with_columns(
            pl.when(
                pl.col.data_date.ge(pl.col.date)
                & pl.col.data_date.le(pl.col.month_end_date)
            )
            .then(
                (pl.col.data_date - pl.col("date"))
                .alias("days_in_month")
                .dt.total_days()
                .cast(pl.Int16())
                + 1
            )
            .otherwise(
                (
                    pl.col("month_end_date")
                    - pl.col("date")
                    + pl.duration(days=1)
                )
                .alias("days_in_month")
                .dt.total_days()
                .cast(pl.Int16())
            )
        )
        .filter(~pl.col.date.gt(pl.col.data_date))
        .drop("month_end_date")
    )

    isr_df = isr_df.join(date_info, on="date").with_columns(
        in_stock_ratio=pl.min_horizontal(
            pl.col.in_stock_days / pl.col.days_in_month, pl.lit(1.0)
        )
    )

    assert (
        len(
            isr_df.with_columns(
                delta_days=pl.max_horizontal(
                    (pl.col.in_stock_days - pl.col.days_in_month), pl.lit(0)
                )
            ).filter(pl.col.delta_days.gt(0))
        )
        == 0
    ), isr_df.with_columns(
        delta_days=pl.max_horizontal(
            (pl.col.in_stock_days - pl.col.days_in_month), pl.lit(0)
        )
    ).filter(pl.col.delta_days.gt(1))

    isr_df = cast_standard(
        [active_sku_info],
        isr_df.rename({"m_sku": "sku"}),
        strict=False,  # skip any SKUs that are not in the active_sku_info list
    ).filter(~(pl.col.sku.is_null() | pl.col.a_sku.is_null()))

    isr_df = isr_df.join(
        active_sku_info.select(
            WHOLE_SKU_IDS + ["sku_year_history", "a_category"]
        ),
        on=["sku", "a_sku"],
    ).rename({"a_category": "category"})

    isr_df = isr_df.select(
        WHOLE_SKU_IDS
        + ["category", "sku_year_history"]
        + Channel.members(MemberType.META)
        + ["date", "in_stock_days", "days_in_month", "in_stock_ratio"]
    )

    write_df(overwrite, isr_path, isr_df)

    return isr_df
