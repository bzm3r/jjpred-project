"""Helpers for checking the health of the data used to make a dispatch, and
comparing it with actual dispatch data."""

from __future__ import annotations

from dataclasses import dataclass, field
import polars as pl
from pathlib import Path

from jjpred.analysisdefn import FbaRevDefn
from jjpred.channel import Channel
from jjpred.datagroups import (
    ALL_IDS,
    ALL_SKU_AND_CHANNEL_IDS,
    ALL_SKU_IDS,
    CHANNEL_IDS,
    CURRENT_PERIOD_INFO,
    DATA_AVAILABILITY_FLAGS,
    DATA_PROBLEM_FLAGS,
    DEMAND_INFO,
    DISPATCH_INFO,
    DISPATCHABLE_PAUSED_DATA,
    FINAL_CHECK_FLAGS,
    INVENTORY_INFO,
    MAIN_PROGRAM_INFO,
    PAUSE_FLAGS,
    PAUSE_PLAN_IDS,
    PREDICTION_MODE_INFO,
    SEASON_IDS,
    WHOLE_SKU_IDS,
)
from jjpred.dispatcher import Dispatcher
from jjpred.globalpaths import ANALYSIS_OUTPUT_FOLDER
from jjpred.readsupport.mainprogram import (
    read_current_period_defn,
    read_excel_predictions,
)
from jjpred.sku import Sku
from jjpred.skuinfo import override_sku_info
from jjpred.structlike import MemberType
from jjpred.utils.fileio import write_excel
from jjpred.utils.polars import (
    NoOverride,
    binary_partition_strict,
    find_dupes,
    join_and_coalesce,
)
from jjpred.utils.typ import PolarsLit


# TODO: make a checker which ensures that no category made a prediction based on
# monthly ratios which are highly concentrated in a single month


@dataclass
class CheckResult:
    """Contains dataframes produced for checking purposes."""

    column_groups: dict[str, list[str]] = field(init=False)
    """Groups of columns that are useful."""

    dfs: dict[str, pl.DataFrame]
    """Keys are the name of the dataframe, which will also be used as the name
    of the Excel sheet which displays the dataframe."""

    def __init__(self, dfs: dict[str, pl.DataFrame], **column_groups):
        self.dfs = dfs
        self.column_groups = column_groups


def coalesce_check_flags(
    df: pl.DataFrame, check_flags: list[str]
) -> pl.DataFrame:
    bool_check_flags = [
        x
        for x in check_flags
        if (x in df.schema.keys()) and (df.schema[x] == pl.Boolean())
    ]
    int_check_flags = [
        x
        for x in check_flags
        if (x in df.schema.keys())
        and (
            df.schema[x]
            in [
                pl.Int8(),
                pl.Int16(),
                pl.Int32(),
                pl.Int64(),
                pl.UInt8(),
                pl.UInt16(),
                pl.UInt32(),
                pl.UInt64(),
            ]
        )
    ]

    collapsed_flags = [f"{x}_str" for x in check_flags if x in df.columns]
    df = (
        df.with_columns(
            pl.when(pl.col(x).is_not_null() & pl.col(x))
            .then(pl.lit(x))
            .otherwise(None)
            .alias(f"{x}_str")
            for x in bool_check_flags
            if x in df.columns
        )
        .with_columns(
            pl.when(pl.col(x).is_not_null() & pl.col(x).gt(0))
            .then(pl.lit(x))
            .otherwise(None)
            .alias(f"{x}_str")
            for x in int_check_flags
            if x in df.columns
        )
        .with_columns(
            check_flags=pl.concat_list(collapsed_flags)
            .list.drop_nulls()
            .list.eval(pl.element())
        )
        .drop(collapsed_flags)
    )

    return df


def check_dispatch_results(
    analysis_defn: FbaRevDefn,
    dispatcher: Dispatcher,
    jjpred_dispatch: pl.DataFrame,
    actual_dispatch: pl.DataFrame | None = None,
    read_from_disk: bool = False,
    main_versus_this_tolerance: int = 0,
) -> CheckResult:
    """Check results calculated by JJPRED program.

    :param analysis_defn: ID of the analysis that the dispatch results correspond to.
    :param mainprogram_date: Date of the main program file to read from, in
        order to get which SKUs are missing from the main program and which
        categories do not have current periods defined properly.
    :param jjpred_dispatch:  Results calculated by JJPRED program.
    :param actual_dispatch: Optional actual dispatch (usually read from a REFILL
        DRAFT PLAN Excel file) to compare against.
    :param main_versus_this_tolerance: If the difference in dispatch for the
      main program vs. this program is equal to or less than this many
      units, then we consider it to be okay.
    :return: Structure containing check-info dataframes.
    """
    active_results = jjpred_dispatch.filter(pl.col("is_active"))

    skus_in_main_program = None
    bad_date_defn = None
    try:
        skus_in_main_program = (
            read_excel_predictions(
                analysis_defn, read_from_disk=read_from_disk
            )
            .select("sku")
            .unique()
            .with_columns(in_main_program=pl.lit(True))
        )

        bad_date_defn = (
            read_current_period_defn(
                analysis_defn, read_from_disk=read_from_disk
            )
            .filter((pl.col("start") - pl.col("end")).eq(0))
            .select("category")
            .with_columns(pl.lit(True).alias("missing_current_period_defn"))
        )

    except (OSError, ValueError) as e:
        print(e)
        skus_in_main_program = None

    if skus_in_main_program is not None and bad_date_defn is not None:
        active_results = override_sku_info(
            active_results,
            skus_in_main_program,
            fill_null_value=PolarsLit(False),
            dupe_check_index=ALL_SKU_AND_CHANNEL_IDS,
        )

        active_results = override_sku_info(
            active_results,
            bad_date_defn,
            fill_null_value=PolarsLit(False),
            dupe_check_index=ALL_SKU_AND_CHANNEL_IDS,
        )

    unpaused_results = active_results.filter(
        ~pl.col("is_master_paused") & ~pl.col("is_config_paused")
    )

    all_sku_ids = ["a_sku"] + Sku.members(MemberType.META)
    sku_ids = Sku.members(MemberType.META)

    if actual_dispatch is not None:
        amazon_inv_check = coalesce_check_flags(
            (
                unpaused_results.join(
                    actual_dispatch.select(
                        ["sku"]
                        + CHANNEL_IDS
                        + [
                            pl.col.final_dispatch.alias(
                                "actual_final_dispatch"
                            )
                        ]
                    ),
                    on=["sku"] + CHANNEL_IDS,
                    how="left",
                )
                .filter(
                    (
                        pl.col.actual_final_dispatch.eq(0)
                        | pl.col.actual_final_dispatch.is_null()
                    )
                    & pl.col.wh_dispatchable.gt(0)
                    & pl.col.ch_stock.eq(0)
                )
                .select(
                    WHOLE_SKU_IDS
                    + CHANNEL_IDS
                    + SEASON_IDS
                    + PAUSE_PLAN_IDS
                    + PAUSE_FLAGS
                    + INVENTORY_INFO
                    + [
                        "prediction_type",
                        "category_marked_new",
                        "refers_to",
                        "referred_by",
                    ]
                    + DATA_AVAILABILITY_FLAGS
                    + ["dispatch", "actual_final_dispatch"]
                )
            ),
            FINAL_CHECK_FLAGS,
        )
    else:
        amazon_inv_check = None

    general_check = {
        "AmazonInvCheck": amazon_inv_check,
        "Dispatchable Paused": active_results.filter(
            (
                (pl.col("is_current_sku") | pl.col("is_future_sku"))
                & (pl.col("is_master_paused") | pl.col("is_config_paused"))
                | (
                    ~(pl.col("is_current_sku") | pl.col("is_future_sku"))
                    & (pl.col("is_master_paused") | pl.col("is_config_paused"))
                )
            )
            & ~pl.col("zero_wh_dispatchable")
            & pl.col.dispatch.eq(0)
        )
        .select(ALL_SKU_IDS + CHANNEL_IDS + DISPATCHABLE_PAUSED_DATA)
        .unique()
        .sort(
            ALL_SKU_IDS + CHANNEL_IDS + ["wh_dispatchable", "expected_demand"]
        ),
        "Missing CONFIG": active_results.filter(
            ~pl.col.in_config_file,
            pl.col.is_current_category,
            ~pl.col.is_master_paused,
        )
        .select(
            "category",
            "category_year_history",
            "is_current_category",
            "is_new_category",
            "in_config_file",
        )
        .unique(),
        "Priorities (No FBA Stock)": active_results.group_by(
            "category", *Channel.members()
        ).agg(
            pl.col.ch_stock.sum(),
            pl.col.is_master_paused.any(),
            pl.col.dispatch.sum(),
        ),
        "NE data missing (sku)": active_results.filter(
            pl.col.uses_ne
            & ~(pl.col.has_po_data | pl.col.has_e_data)
            & pl.col.is_current_sku
        )
        .select(ALL_IDS + DATA_AVAILABILITY_FLAGS)
        .unique()
        .sort(ALL_SKU_AND_CHANNEL_IDS),
        "NE data missing (cat)": active_results.filter(
            pl.col.uses_ne
            & ~(pl.col.has_po_data | pl.col.has_e_data)
            & pl.col.is_current_sku
        )
        .select("category")
        .unique()
        .sort("category"),
        "PO data missing (sku)": active_results.filter(
            pl.col("po_problem")
            & ~pl.col("is_future_sku")
            & pl.col("wh_dispatchable").gt(0)
        )
        .select(
            ALL_IDS
            + [
                "has_po_data",
                "wh_dispatchable",
            ]
        )
        .unique()
        .sort(all_sku_ids),
        "PO data missing (cat)": active_results.filter(
            pl.col("po_problem")
            & ~pl.col("is_future_sku")
            & pl.col("wh_dispatchable").gt(0)
        )
        .select("category")
        .unique()
        .sort("category"),
        "E new sku check": active_results.filter(
            pl.col("prediction_type").eq("E")
            & ~(pl.col("is_master_paused") | pl.col("is_config_paused"))
            & (pl.col("is_new_sku") | pl.col("is_new_category"))
            & pl.col("wh_dispatchable").gt(0)
            & pl.col("ch_stock").lt(3)
            & pl.col("dispatch_below_cutoff")
        ).select(
            ALL_IDS
            + DATA_PROBLEM_FLAGS
            + PREDICTION_MODE_INFO
            + INVENTORY_INFO
            + CURRENT_PERIOD_INFO
            + [
                "dispatch",
            ]
        ),
        "E data missing": active_results.filter(
            pl.col("e_problem")
            & (~pl.col("is_future_sku") | pl.col("wh_dispatchable").gt(0))
        )
        .select(
            ALL_IDS
            + PAUSE_FLAGS
            + [x for x in DATA_AVAILABILITY_FLAGS if x != "has_po_data"]
            + INVENTORY_INFO
            + PREDICTION_MODE_INFO
        )
        .unique()
        .sort(ALL_SKU_AND_CHANNEL_IDS),
        "Main program missing": active_results.filter(
            ~pl.col("in_main_program") & pl.col("is_current_sku")
            | (
                ~pl.col("in_main_program")
                & ~(pl.col("is_current_sku") | pl.col("is_future_sku"))
                & pl.col("wh_dispatchable").gt(0)
            )
        )
        .select(
            ALL_SKU_IDS
            + SEASON_IDS
            + PAUSE_FLAGS
            + PAUSE_PLAN_IDS
            + MAIN_PROGRAM_INFO
            + ["wh_stock"]
        )
        .unique()
        .sort(all_sku_ids)
        if "in_main_program" in active_results
        else pl.DataFrame(),
        "Missing Current Period": unpaused_results.filter(
            pl.col("missing_current_period_defn") & pl.col("is_current_sku")
        )
        .select(
            ["category"]
            + SEASON_IDS
            + [
                "missing_current_period_defn",
                "current_period",
            ]
        )
        .unique()
        .sort(["category"])
        if "missing_current_period_defn" in unpaused_results
        else pl.DataFrame(),
        "AMZ INV missing": unpaused_results.filter(
            pl.col("no_ch_stock_info")
            & (
                pl.col("is_current_sku")
                | (~pl.col("is_future_sku") & pl.col("wh_dispatchable").gt(0))
            )
        )
        .select(
            ALL_SKU_AND_CHANNEL_IDS
            + SEASON_IDS
            + [
                "no_ch_stock_info",
                "wh_dispatchable",
            ]
        )
        .unique()
        .sort(ALL_SKU_AND_CHANNEL_IDS)
        if "no_ch_stock_info" in unpaused_results
        else pl.DataFrame(),
        "WH INV missing": unpaused_results.filter(
            pl.col("no_wh_stock_info") & pl.col("is_current_sku")
        )
        .select(
            ALL_SKU_IDS
            + SEASON_IDS
            + [
                "no_wh_stock_info",
            ]
        )
        .unique()
        .sort(all_sku_ids),
    }

    # Collapse boolean check flags into a comma separated list of strings, in
    # order to make it easy to read in Excel.
    results_with_check_flags = coalesce_check_flags(
        jjpred_dispatch.with_columns(
            low_current_period_sales=pl.when(
                pl.col.low_current_period_sales
                & pl.col.demand_based_on_e.gt(0)
            )
            .then(pl.col.low_current_period_sales)
            .otherwise(False),
            low_category_historical_sales=pl.when(
                pl.col.low_category_historical_sales
                & pl.col.demand_based_on_e.gt(0)
            )
            .then(pl.col.low_category_historical_sales)
            .otherwise(False),
        ).select(
            x
            for x in (
                ALL_IDS
                + PREDICTION_MODE_INFO
                + DEMAND_INFO
                + DISPATCH_INFO
                + FINAL_CHECK_FLAGS
            )
            if x in jjpred_dispatch.columns
        ),
        FINAL_CHECK_FLAGS,
    )

    # Generate checks involving comparison between actual dispatch and
    # program generated dispatch results.
    actual_dispatch_gt_wh_dispatchable = None
    not_in_jjpred = None
    diff_low_isr = None
    not_in_actual = None
    diff_wrt_actual = None
    ok_wrt_actual = None
    ok_no_dispatch = None
    ok_dispatch = None
    jjpred_dispatch = results_with_check_flags.filter(
        ~pl.col("dispatch_below_cutoff")
    )
    contained_check_flags = [
        x for x in FINAL_CHECK_FLAGS if x in results_with_check_flags.columns
    ]
    if actual_dispatch is not None:
        actual_dispatch = actual_dispatch.rename(
            {
                "final_dispatch": "actual_final_dispatch",
                "calc_dispatch": "actual_calc_dispatch",
            }
        )
        # shared_id_cols = [x for x in ALL_IDS if x in actual_dispatch.columns]

        find_dupes(
            actual_dispatch, ["sku"] + Channel.members(), raise_error=True
        )

        with_actual_dispatch = (
            join_and_coalesce(
                results_with_check_flags,
                actual_dispatch,
                NoOverride(),
                nulls_equal=True,
                dupe_check_index=ALL_SKU_AND_CHANNEL_IDS,
            )
            .with_columns(
                pl.col.dispatch.fill_null(0),
                pl.col.actual_final_dispatch.fill_null(0),
                pl.col.manually_deleted.fill_null(pl.lit(False)),
                pl.col.manually_altered.fill_null(pl.lit(False)),
                pl.col.actual_calc_dispatch.fill_null(0),
            )
            .with_columns(
                dispatch_delta=pl.col("dispatch")
                - pl.col("actual_final_dispatch")
            )
            .sort(pl.col("dispatch_delta").abs())
            .with_columns(
                within_tolerance=(
                    pl.col("dispatch_delta")
                    .abs()
                    .le(main_versus_this_tolerance)
                ),
            )
        )
        ok_wrt_actual, with_actual_dispatch = binary_partition_strict(
            with_actual_dispatch,
            pl.col.within_tolerance
            | (
                pl.col.dispatch_below_cutoff
                & pl.col.actual_final_dispatch.eq(0)
            ),
        )
        ok_no_dispatch, ok_dispatch = binary_partition_strict(
            ok_wrt_actual,
            (
                pl.col.dispatch_below_cutoff
                & pl.col.actual_final_dispatch.eq(0)
            ),
        )
        actual_dispatch_gt_wh_dispatchable, with_actual_dispatch = (
            binary_partition_strict(
                with_actual_dispatch,
                pl.col.actual_final_dispatch.gt(pl.col.wh_dispatchable),
            )
        )
        # diff_low_isr, with_actual_dispatch = binary_partition_strict(
        #     with_actual_dispatch,
        #     # (
        #     #     (
        #     #         pl.col.actual_final_dispatch.eq(0)
        #     #         & pl.col.dispatch.gt(0)
        #     #         & ~pl.col.within_tolerance
        #     #     )
        #     #     | (~pl.col.within_tolerance)
        #     # )
        #     # & pl.col.has_low_isr.sum().gt(0),
        #     pl.col.has_low_isr.sum().gt(0) & ~pl.col.dispatch_below_cutoff,
        # )
        not_in_jjpred, with_actual_dispatch = binary_partition_strict(
            with_actual_dispatch,
            pl.col.actual_final_dispatch.ge(analysis_defn.dispatch_cutoff_qty)
            & (pl.col.dispatch_below_cutoff | pl.col.dispatch.is_null()),
        )
        not_in_actual, with_actual_dispatch = binary_partition_strict(
            with_actual_dispatch,
            pl.col.actual_final_dispatch.lt(analysis_defn.dispatch_cutoff_qty)
            & ~(pl.col.dispatch_below_cutoff | pl.col.dispatch.is_null()),
        )
        diff_wrt_actual = with_actual_dispatch

    multi_a_sku_df = (
        find_dupes(
            active_results.select(
                ["sku", "a_sku"]
                + Channel.members()
                + ["pause_plan", "dispatch"]
            )
            .with_columns(
                paused=(pl.col.country_flag & pl.col.pause_plan).gt(0)
            )
            .filter(~pl.col.paused),
            ["a_sku"] + Channel.members(),
        )
        .with_columns(
            attention_required=pl.col.dispatch.list.eval(
                pl.element().gt(0)
            ).list.any()
        )
        .sort("attention_required", descending=True)
    )

    assert results_with_check_flags["expected_demand"].dtype == pl.Int64()
    dispatch_checks = {
        "(wh_lt_actual) MainProg|JJPRED": actual_dispatch_gt_wh_dispatchable.drop(
            contained_check_flags
        )
        if actual_dispatch_gt_wh_dispatchable is not None
        else None,
        "(miss JJPRED) MainProg|JJPRED": not_in_jjpred.drop(
            contained_check_flags
        )
        if not_in_jjpred is not None
        else None,
        "(miss MainProg) MainProg|JJPRED": not_in_actual.drop(
            contained_check_flags
        )
        if not_in_actual is not None
        else None,
        "(diff) MainProg|JJPRED": diff_wrt_actual.drop(contained_check_flags)
        if diff_wrt_actual is not None
        else None,
        "(low isr) MainProg|JJPRED": diff_low_isr.drop(contained_check_flags)
        if diff_low_isr is not None
        else None,
        # "(cutoff) MainProg|JJPRED": actual_vs_jjpred.filter(
        #     pl.col("dispatch_below_cutoff")
        #     & pl.col("actual_final_dispatch").gt(0)
        # ).drop(contained_check_flags)
        # if actual_vs_jjpred is not None
        # else None,
        "(ok) no dispatch": ok_no_dispatch.drop(contained_check_flags)
        if ok_no_dispatch is not None
        else None,
        "(ok) MainProg|JJPRED": ok_dispatch.drop(contained_check_flags)
        if ok_dispatch is not None
        else None,
        "Current Period": active_results.select(
            "category", "season", "current_period"
        )
        .unique()
        .sort("season", "category"),
        "Reference Category": active_results.select(
            "season",
            "category",
            "category_year_history",
            "category_type",
            "refers_to",
            "referred_by",
        )
        .unique()
        .sort("season", "category"),
        "Low Demand Prints": results_with_check_flags.filter(
            ~(pl.col("is_master_paused") | pl.col("is_config_paused"))
            & pl.col("expected_demand").lt(2)
            & pl.col("wh_dispatchable").gt(0)
            & pl.col("ch_stock").eq(0)
        )
        .drop(contained_check_flags)
        .unique()
        .sort(ALL_SKU_AND_CHANNEL_IDS),
        "PO vs E check": results_with_check_flags.filter(
            ~(pl.col("is_master_paused") | pl.col("is_config_paused"))
        )
        .with_columns(
            expected_demand_diff=pl.col("expected_demand_from_po").list.sum()
            - pl.col("expected_demand_from_history").list.sum()
        )
        .drop(contained_check_flags)
        .unique()
        .sort(ALL_SKU_AND_CHANNEL_IDS),
        "ASG-GRA check": results_with_check_flags.filter(
            pl.col.print.eq("ASG")
        )
        .drop(contained_check_flags)
        .unique()
        .sort(ALL_SKU_AND_CHANNEL_IDS),
        "all": active_results,
        "box rounding": active_results.filter(
            pl.col.qty_box.is_not_null() & pl.col.qty_box.gt(0),
            pl.col.requesting.gt(pl.col.ch_stock),
        ).select(
            "sku",
            "platform",
            "country_flag",
            "ch_stock",
            "wh_stock",
            "expected_demand",
            "requesting",
            "pre_box_required",
            "qty_box",
            "exact_boxes",
            "exact_boxes_floor",
            "exact_boxes_ceil",
            "qty_exact_boxes_floor",
            "qty_exact_boxes_ceil",
            "distance_to_floor",
            "distance_to_ceil",
            "margin_by_ratio",
            "close_to_floor",
            "close_to_ceil",
            "num_closest_box",
            "post_box_required",
            "post_split_required",
            "post_split_exact_boxes",
            "post_split_exact_boxes_floor",
            "post_split_distance_to_floor",
            "post_split_margin_by_ratio",
            "eb_not_null",
            "close_by_ratio",
            "close_by_qty",
            "eb_gt_zero",
            "post_split_close_to_floor",
            "post_split_num_closest_box",
            (
                pl.when(pl.col.auto_split)
                .then(pl.col.post_split_num_closest_box.is_not_null())
                .otherwise(pl.col.num_closest_box.is_not_null())
            ).alias("rounded_to_box"),
            "dispatch",
            "auto_split",
        ),
        "estimation_check": active_results.select(
            "sku",
            "category",
            "season",
            "sku_year_history",
            "category_year_history",
            "is_master_paused",
            "is_config_paused",
            (pl.col.is_master_paused | pl.col.is_config_paused).alias(
                "is_paused"
            ),
            pl.concat_arr(
                *[pl.col(x).cast(pl.String()) for x in Channel.members()]
            ).arr.join(" "),
            "ch_stock",
            "wh_stock",
            "prediction_type",
            "uses_ce",
            "ce_uses_e",
            "ce_uses_po",
            "pre_box_required",
            pl.col("expected_demand_from_history").list.sum(),
            pl.col("expected_demand_from_po").list.sum(),
            "dispatch",
        ).with_columns(
            (
                pl.col.expected_demand_from_history
                < 0.5 * pl.col.expected_demand_from_po
            ).alias("e < 0.5 * po"),
            (
                pl.col.expected_demand_from_po
                - pl.col.expected_demand_from_history
            ).alias("po - e"),
        ),
        "reserved": dispatcher.reserved_quantity,
        "expected_demand_last_year": dispatcher.expected_demand_last_year,
        "reserved_conflict_sku": (
            active_results.filter(pl.col.reserved.gt(0))
            .select(
                "sku",
                *Channel.members(),
                "category",
                "wh_stock",
                "wh_on_order",
                "reserved_before_on_order",
                "reserved",
                "wh_dispatchable",
                "post_box_required",
                "total_post_box_required",
                "dispatch",
            )
            .sort("sku")
            .with_columns(
                total_possible_dispatch=pl.min_horizontal(
                    pl.col.total_post_box_required,
                    pl.col.wh_dispatchable + pl.col.reserved,
                )
            )
            .filter(pl.col.total_possible_dispatch.gt(0))
            .with_columns(
                total_dispatch=pl.col.dispatch.sum().over("sku"),
            )
            .with_columns(
                reserve_percent=pl.when(
                    pl.col.wh_stock.eq(0) | pl.col.wh_stock.lt(pl.col.reserved)
                )
                .then(1.0)
                .otherwise(pl.col.reserved / pl.col.wh_stock),
            )
            .with_columns(
                total_dispatch_over_total_possible_dispatch=pl.col.total_dispatch
                / pl.col.total_possible_dispatch,
            )
            .sort("total_dispatch_over_total_possible_dispatch")
            .filter(~pl.col.total_dispatch.ge(pl.col.total_possible_dispatch))
        ),
        "reserved_conflict_category": (
            active_results.filter(pl.col.reserved.gt(0))
            .select(
                "sku",
                *Channel.members(),
                "category",
                "wh_stock",
                "wh_on_order",
                "reserved_before_on_order",
                "reserved",
                "wh_dispatchable",
                "post_box_required",
                "total_post_box_required",
                "dispatch",
            )
            .sort("sku")
            .with_columns(
                total_possible_dispatch=pl.min_horizontal(
                    pl.col.total_post_box_required,
                    pl.col.wh_dispatchable + pl.col.reserved,
                )
            )
            .filter(pl.col.total_possible_dispatch.gt(0))
            .with_columns(
                total_dispatch=pl.col.dispatch.sum().over("sku"),
            )
            .group_by("category", *Channel.members())
            .agg(
                pl.col.wh_stock.sum(),
                pl.col.wh_on_order.sum(),
                pl.col.reserved_before_on_order.sum(),
                pl.col.reserved.sum(),
                pl.col.wh_dispatchable.sum(),
                pl.col.post_box_required.sum(),
                pl.col.total_post_box_required.sum(),
                pl.col.dispatch.sum(),
                pl.col.total_dispatch.sum(),
                pl.col.total_possible_dispatch.sum(),
            )
            .with_columns(
                reserve_percent=pl.when(
                    pl.col.wh_stock.eq(0) | pl.col.wh_stock.lt(pl.col.reserved)
                )
                .then(1.0)
                .otherwise(pl.col.reserved / pl.col.wh_stock),
            )
            .with_columns(
                total_dispatch_over_total_possible_dispatch=pl.col.total_dispatch
                / pl.col.total_possible_dispatch,
            )
            .sort("total_dispatch_over_total_possible_dispatch")
            .filter(~pl.col.total_dispatch.ge(pl.col.total_possible_dispatch))
        ),
        "reserve_percent_sku": active_results.filter(pl.col.reserved.gt(0))
        .select(
            "sku",
            "category",
            "wh_stock",
            "wh_on_order",
            "reserved_before_on_order",
            "reserved",
        )
        .unique()
        .sort("sku")
        .with_columns(
            reserve_percent=pl.when(
                pl.col.wh_stock.eq(0) | pl.col.wh_stock.lt(pl.col.reserved)
            )
            .then(1.0)
            .otherwise(pl.col.reserved / pl.col.wh_stock)
        )
        .sort("reserved", descending=True),
        "reserve_percent_category": active_results.filter(
            pl.col.reserved.gt(0)
        )
        .select(
            "sku",
            "category",
            "wh_stock",
            "wh_on_order",
            "reserved_before_on_order",
            "reserved",
        )
        .unique()
        .group_by("category")
        .agg(
            pl.col.wh_stock.sum(),
            pl.col.wh_on_order.sum(),
            pl.col.reserved_before_on_order.sum(),
            pl.col.reserved.sum(),
        )
        .with_columns(
            reserve_percent=pl.when(
                pl.col.wh_stock.eq(0) | pl.col.wh_stock.lt(pl.col.reserved)
            )
            .then(1.0)
            .otherwise(pl.col.reserved / pl.col.wh_stock)
        )
        .sort("reserved", descending=True),
        "multi_a_sku": multi_a_sku_df,
    }

    # now = datetime.datetime.now().strftime("%Y-%b-%d_%H%M%S")
    result_path = ANALYSIS_OUTPUT_FOLDER.joinpath(
        Path(f"check_results_{analysis_defn.tag_with_output_time()}.xlsx")
    )
    write_excel(result_path, (general_check | dispatch_checks))
    # print(f"Saving check results to: {result_path}")

    # if result_path.exists():
    #     result_path.unlink()

    # with xlw.Workbook(result_path) as workbook:
    #     for key, df in (general_check | dispatch_checks).items():
    #         if df is not None:
    #             convert_df_for_excel(df).write_excel(
    #                 workbook=workbook, worksheet=key
    #             )

    return CheckResult(
        (
            dict(
                (k, v)
                for k, v in (
                    general_check | dispatch_checks
                    # | {"actual_vs_jj": with_actual_dispatch}
                ).items()
                if v is not None
            )
        ),
        all_sku_ids=all_sku_ids,
        sku_ids=sku_ids,
        ALL_IDS=ALL_IDS,
        check_flags=FINAL_CHECK_FLAGS,
        prediction_mode_info=PREDICTION_MODE_INFO,
        demand_info=DEMAND_INFO,
        dispatch_info=DISPATCH_INFO,
    )
