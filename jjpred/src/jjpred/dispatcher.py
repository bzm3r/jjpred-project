"""Logic for calculating a dispatch for FBA refill."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import polars as pl

from analysis_tools.utils import get_analysis_defn_and_db
from jjpred.analysisdefn import RefillDefn
from jjpred.channel import Channel, DistributionMode, Platform
from jjpred.datagroups import (
    ALL_IDS,
    ALL_SKU_AND_CHANNEL_IDS,
    ALL_SKU_IDS,
    CHANNEL_IDS,
    MASTER_PAUSE_FLAGS,
    PAUSE_PLAN_IDS,
    NOVELTY_FLAGS,
    SEASON_IDS,
    STATUS_IDS,
    WHOLE_SKU_IDS,
)
from jjpred.countryflags import CountryFlags
from jjpred.database import DataBase
from jjpred.dispatchformatter import (
    format_fba_dispatch_for_netsuite,
    format_jjweb_dispatch_for_netsuite,
)
from jjpred.globalpaths import ANALYSIS_OUTPUT_FOLDER
from jjpred.predictiontypes import PredictionType
from jjpred.predictor import Predictor
from jjpred.readsheet import DataVariant
from jjpred.readsupport.qtybox import read_qty_box
from jjpred.readsupport.marketing import ConfigData, read_config
from jjpred.readsupport.utils import cast_standard
from jjpred.sku import Sku
from jjpred.skuinfo import (
    override_sku_info,
    attach_channel_info,
    attach_refill_info_from_config,
)
from jjpred.strategies import ChannelStrategyGroups
from jjpred.strategylib import (
    LATEST,
    get_last_year_as_current_period_dict,
    get_strategy_from_library,
)
from jjpred.structlike import MemberType, StructLike
from jjpred.utils.datetime import Date, DateLike
from jjpred.utils.fileio import write_df, write_excel
from jjpred.utils.polars import (
    FilterStructs,
    binary_partition_strict,
    concat_enum_extend_vstack_strict,
    find_dupes,
    struct_filter,
)
from jjpred.utils.typ import PolarsLit, normalize_optional


def calculate_required(
    analysis_defn: RefillDefn,
    predicted_demand: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate the quantity of goods required by a channel, given its
    predicted demand.

    This quantity depends on:
    * the current inventory of the channel (we subtract the current inventory
    from the predicted demand quantity to get the ``pre_box_required``
    (pre-full-box logic required amount)
    * we round ``pre_box_required/qty_box`` to get how many boxes of goods
    are required (``exact_boxes``)

      * if ``pre_box_required`` is within 10% of this quantity, then we
        use ``exact_boxes * qty_box`` as the ``post_box_required`` amount for the
        channel
      * otherwise we ``pre_box_required`` as the ``post_box_required`` amount
    """
    df = predicted_demand

    assert df["expected_demand"].dtype == pl.Int64()
    df = df.with_columns(
        requesting=pl.max_horizontal(
            pl.col.expected_demand.ceil().cast(pl.Int64()),
            pl.col.refill_request,
        )
    ).with_columns(
        uses_refill_request=(
            pl.col.requesting.eq(pl.col.refill_request)
            & pl.col.refill_request.gt(0)
        )
    )

    df = df.with_columns(
        pre_box_required=pl.when(pl.col("requesting") > pl.col("ch_stock"))
        .then(pl.col("requesting") - pl.col("ch_stock"))
        .otherwise(0)
    )

    # IHT-DPK-L
    df = (
        df.with_columns(
            enable_full_box_logic=pl.lit(
                analysis_defn.enable_full_box_logic, dtype=pl.Boolean()
            )
        )
        .with_columns(
            exact_boxes=pl.when(
                pl.col.enable_full_box_logic
                & pl.col.qty_box.is_not_null()
                & pl.col.qty_box.gt(0)
            ).then(pl.col.pre_box_required / pl.col.qty_box)
        )
        .with_columns(
            exact_boxes_floor=pl.col.exact_boxes.floor().cast(pl.Int64()),
            exact_boxes_ceil=pl.col.exact_boxes.ceil().cast(pl.Int64()),
        )
        .with_columns(
            qty_exact_boxes_floor=(
                pl.col.qty_box * pl.col.exact_boxes_floor
            ).cast(pl.Int64),
            qty_exact_boxes_ceil=(
                pl.col.qty_box * pl.col.exact_boxes_ceil
            ).cast(pl.Int64()),
        )
        .with_columns(
            distance_to_floor=(
                pl.col.pre_box_required.sub(pl.col.qty_exact_boxes_floor)
            ),
            distance_to_ceil=pl.col.qty_exact_boxes_ceil.sub(
                pl.col.pre_box_required
            ),
            margin_by_ratio=(
                analysis_defn.full_box_rounding_margin_ratio
                * pl.col.pre_box_required
            )
            .ceil()
            .cast(pl.Int64()),
        )
        .with_columns(
            close_to_floor=(
                pl.col.exact_boxes.is_not_null()
                & (
                    pl.col.distance_to_floor.le(pl.col.margin_by_ratio)
                    | pl.col.distance_to_floor.le(
                        pl.lit(analysis_defn.full_box_rounding_margin_qty)
                    )
                )
                & pl.col.exact_boxes_floor.gt(0)
            ),
            close_to_ceil=(
                pl.col.exact_boxes.is_not_null()
                & (
                    pl.col.distance_to_ceil.le(pl.col.margin_by_ratio)
                    | pl.col.distance_to_ceil.le(
                        pl.lit(analysis_defn.full_box_rounding_margin_qty)
                    )
                )
            ),
        )
        .with_columns(
            num_closest_box=pl.when(pl.col.exact_boxes.is_not_null()).then(
                pl.when(pl.col.close_to_floor & pl.col.close_to_ceil)
                .then(pl.col.exact_boxes_ceil)
                .when(
                    pl.col.close_to_floor.xor(pl.col.close_to_ceil)
                    & pl.col.close_to_floor
                )
                .then(pl.col.exact_boxes_floor)
                .when(
                    pl.col.close_to_floor.xor(pl.col.close_to_ceil)
                    & pl.col.close_to_ceil
                )
                .then(pl.col.exact_boxes_ceil)
                .otherwise(pl.lit(None))
            ),
            post_box_required=pl.when(pl.col.exact_boxes.is_not_null()).then(
                pl.when(pl.col.close_to_floor & pl.col.close_to_ceil)
                .then(pl.col.qty_exact_boxes_ceil)
                .when(
                    pl.col.close_to_floor.xor(pl.col.close_to_ceil)
                    & pl.col.close_to_floor
                )
                .then(pl.col.qty_exact_boxes_floor)
                .when(
                    pl.col.close_to_floor.xor(pl.col.close_to_ceil)
                    & pl.col.close_to_ceil
                )
                .then(pl.col.qty_exact_boxes_ceil)
                .otherwise(pl.col.pre_box_required)
            ),
        )
    )

    df = df.with_columns(
        total_post_box_required=pl.col("post_box_required")
        .sum()
        .over(
            [
                x
                for x in df.columns
                if (x in ["a_sku"] + Sku.members(MemberType.META))
            ]
        ),
        total_pre_box_required=pl.col("pre_box_required")
        .sum()
        .over(
            [
                x
                for x in df.columns
                if (x in ["a_sku"] + Sku.members(MemberType.META))
            ]
        ),
    )

    assert df["post_box_required"].dtype == pl.Int64()

    return df


def determine_reserve_period_start_end_date(
    db: DataBase,
    dispatch_start_date: DateLike,
    jjweb_reserve_defn: pl.Expr | None,
) -> pl.DataFrame:
    assert isinstance(db.analysis_defn, RefillDefn)

    reserve_period_calculation_df = (
        (
            db.meta_info.active_sku.select("category", "season")
            .unique()
            .with_columns(
                reserve_months=jjweb_reserve_defn.cast(
                    pl.List(pl.Array(pl.UInt8(), 2))
                )
                if jjweb_reserve_defn is not None
                else pl.lit([]).cast(pl.List(pl.Array(pl.UInt8(), 2))),
            )
        )
        .explode("reserve_months")
        .with_columns(
            start=pl.col.reserve_months.arr.first(),
            end=pl.col.reserve_months.arr.last(),
        )
        .select(
            "category",
            "season",
            pl.struct("start", "end").alias("reserve_period"),
        )
        .group_by("category", "season")
        .agg(pl.col.reserve_period.alias("reserve_periods"))
    )

    reserve_period_date_df = (
        reserve_period_calculation_df.with_columns(
            start_date=Date.from_datelike(dispatch_start_date).as_polars_date()
        )
        .explode("reserve_periods")
        .rename({"reserve_periods": "reserve_period"})
        .with_columns(
            start_date_month_ge_period_start=pl.col.start_date.dt.month().ge(
                pl.col.reserve_period.struct.field("start")
            ),
        )
        .filter(pl.col.start_date_month_ge_period_start)
        .with_columns(
            end_date=pl.date(
                pl.when(
                    pl.col.reserve_period.struct.field("end").lt(
                        pl.col.reserve_period.struct.field("start")
                    )
                )
                .then(pl.col.start_date.dt.year() + 1)
                .otherwise(pl.col.start_date.dt.year()),
                pl.col.reserve_period.struct.field("end"),
                1,
            )
        )
        .filter(pl.col.start_date.lt(pl.col.end_date))
    )

    find_dupes(reserve_period_date_df, ["category"], raise_error=True)

    return reserve_period_date_df


def calculate_reserved_quantity(
    db: DataBase,
    predictor: Predictor,
    jjweb_reserve_defn: pl.Expr | None,
    force_po_predictions: bool = True,
) -> pl.DataFrame:
    assert isinstance(db.analysis_defn, RefillDefn)
    find_dupes(db.meta_info.active_sku, ["sku", "a_sku"], raise_error=True)

    end_date_dicts = (
        determine_reserve_period_start_end_date(
            db, db.analysis_defn.dispatch_date, jjweb_reserve_defn
        )
        .group_by("start_date", "end_date")
        .agg(pl.col.category)
        .to_dicts()
    )

    reserve_demands = []

    for end_date_info in end_date_dicts:
        reserve_demand = (
            predictor.predict_demand(
                ["janandjul.com"],
                db.analysis_defn.dispatch_date,
                end_date_info["end_date"],
                force_po_prediction=force_po_predictions,
                aggregate_final_result=True,
            )
            .join(
                db.meta_info.active_sku.select(
                    "a_sku", "sku", "category", "is_current_sku"
                ),
                on=["a_sku", "sku"],
            )
            .filter(pl.col.category.is_in(end_date_info["category"]))
        )

        reserve_demand = reserve_demand.with_columns(
            reserved=pl.when(~pl.col.is_current_sku)
            .then(0)
            .otherwise(pl.col.expected_demand)
        )

        reserve_demands.append(reserve_demand)

    if len(reserve_demands) > 0:
        return concat_enum_extend_vstack_strict(reserve_demands)
    else:
        return pl.DataFrame()


def calculate_jjweb_past_one_year_quantity(
    db: DataBase,
    predictor: Predictor,
) -> pl.DataFrame:
    assert isinstance(db.analysis_defn, RefillDefn)
    find_dupes(db.meta_info.active_sku, ["sku", "a_sku"], raise_error=True)

    predictor = Predictor(
        db.analysis_defn,
        db,
        ChannelStrategyGroups(
            db.analysis_defn,
            input_strategies=get_strategy_from_library(
                db.analysis_defn,
                LATEST,
                current_period_overrides=get_last_year_as_current_period_dict(
                    db
                ),
            ),
        ),
        predictor.po_data,
    )

    predicted_demand = predictor.predict_demand(
        ["jjweb ca east"],
        db.analysis_defn.dispatch_date,
        db.analysis_defn.end_date,
        aggregate_final_result=True,
        force_e_prediction=True,
    )

    return predicted_demand


def attach_inventory_info(
    db: DataBase,
    all_sku_info: pl.DataFrame,
    warehouse_filter: pl.Expr,
    warehouse_min_keep_qty: int,
    filters: FilterStructs | None = None,
) -> pl.DataFrame:
    inv_df = struct_filter(db.dfs[DataVariant.Inventory], filters)
    wh_stock, ch_stock = binary_partition_strict(
        inv_df.select(WHOLE_SKU_IDS + CHANNEL_IDS + ["stock", "on_order"]),
        warehouse_filter,
    )
    wh_stock = (
        wh_stock.filter(
            pl.col("country_flag").eq(int(CountryFlags.CA)),
        )
        .drop(Channel.members())
        .rename({"stock": "wh_stock", "on_order": "wh_on_order"})
        .select(WHOLE_SKU_IDS + ["wh_stock", "wh_on_order"])
    )
    ch_stock = ch_stock.drop("on_order").rename({"stock": "ch_stock"})

    all_sku_info = override_sku_info(
        all_sku_info,
        wh_stock,
        fill_null_value=PolarsLit(0),
        create_info_columns=[
            "wh_stock",
            (
                pl.col("wh_stock").is_not_null() & pl.col("wh_stock").lt(0)
            ).alias("negative_wh_stock"),
        ],
        dupe_check_index=ALL_SKU_AND_CHANNEL_IDS,
    )

    all_sku_info = override_sku_info(
        all_sku_info,
        ch_stock,
        fill_null_value=PolarsLit(0),
        create_info_columns=[
            "ch_stock",
            (
                (
                    ~(
                        pl.col("ch_stock").is_not_null()
                        & pl.col("ch_stock").gt(0)
                    )
                ).alias("zero_ch_stock")
            ),
        ],
    )

    all_sku_info = all_sku_info.with_columns(
        min_keep=pl.lit(warehouse_min_keep_qty, pl.Int64())
    )

    return all_sku_info


# FINE AUTOSPLIT LOGIC: removed for now after discussion with Matt
# This was meant to make sure that auto-splits do not spill into the
# minimum quantity meant to be kept in a warehouse.
# if (
#     requires_fine_splitting is not None
#     or len(requires_fine_splitting) > 0
# ):
#     # the fundamental assumption here is that at this point, the
#     # delta between supply and total required cannot be greater than
#     # the number of SKUs * the number of channels = N
#     #
#     # for those cases where a fix is still required, we rank the
#     # SKUs by their dispatch fraction, and then subtract one from
#     # the dispatch of the bottom K SKUs, where 0 < K <= N.
#     requires_fine_splitting = (
#         # sort the data by Sku, to make the ranking performed
#         # deterministic
#         requires_fine_splitting.sort(
#             Sku.members(MemberType.META)
#         ).with_columns(
#             delta_wh=(
#                 pl.col("total_dispatch")
#                 - pl.col("wh_dispatchable")
#             ),
#             # rank the data for each SKU by size of dispatch
#             # resolve tie-breaks by the order of SKU in the group
#             # (groups are per channel)
#             index=(
#                 pl.col("fraction_dispatch")
#                 .rank("ordinal")
#                 .over(Sku.members())
#                 - 1
#             ),
#         )
#     ).with_columns(
#         # subtract -1 from the bottom "delta_wh" SKUs by
#         # dispatch fraction
#         fixed_dispatch=pl.when(
#             (pl.col("index") < pl.col("delta_wh"))
#             & (pl.col("dispatch").gt(0))
#         )
#         .then(pl.col("dispatch") - 1)
#         .otherwise(pl.col("dispatch"))
#     )
#     # setup for checking that now we have fixed the total dispatch
#     requires_fine_splitting = requires_fine_splitting.with_columns(
#         fixed_total_dispatch=pl.col("fixed_dispatch")
#         .sum()
#         .over(Sku.members(MemberType.META))
#     ).with_columns(
#         fixed_delta_inv=(
#             pl.col("fixed_total_dispatch")
#             - pl.col("wh_dispatchable")
#         )
#     )
#     # fixed_delta_inv should now be 0 across all entries
#     assert (
#         requires_fine_splitting["fixed_delta_inv"].min() == 0
#         and requires_fine_splitting["fixed_delta_inv"].max() == 0
#     ), requires_fine_splitting.filter(
#         ~pl.col("fixed_delta_inv").eq(0)
#     )

#     requires_fine_splitting = requires_fine_splitting.with_columns(
#         dispatch=pl.col("fixed_dispatch"),
#         auto_split=pl.lit(True),
#         fine_auto_split=pl.lit(True),
#     )
#     assert requires_fine_splitting["dispatch"].dtype == pl.Int64()

#     self.all_sku_info = override_sku_info(
#         self.all_sku_info.with_columns(
#             fine_auto_split=pl.lit(False)
#         ),
#         requires_fine_splitting.select(
#             Sku.members(MemberType.META)
#             + Channel.members()
#             + [
#                 "pre_box_required",
#                 "dispatch",
#                 "num_closest_box",
#                 "auto_split",
#                 "fine_auto_split",
#             ]
#         ),
#         fill_null_value=None,
#         create_info_columns=None,
#         create_missing_info_flags=False,
#     ).with_columns(pl.col.auto_split.fill_null(False))


def calculate_full_box_and_auto_split(
    analysis_defn: RefillDefn,
    all_sku_info: pl.DataFrame,
) -> pl.DataFrame:
    # calculated what each channel + SKU requires for dispatch
    required_df = calculate_required(
        analysis_defn,
        all_sku_info.filter(
            pl.col("is_active")
            & ~pl.col("is_master_paused")
            & ~pl.col("is_config_paused")
            & ~pl.col("no_wh_stock_info")
            & ~pl.col("zero_wh_dispatchable")
        ),
    )

    all_sku_info = cast_standard(
        [all_sku_info],
        override_sku_info(
            all_sku_info,
            required_df,
            fill_null_value=defaultdict(
                lambda: PolarsLit(0),
                {
                    k: PolarsLit(False)
                    for k in [
                        "uses_refill_request",
                        "enable_full_box_logic",
                        "close_to_floor",
                        "close_to_ceil",
                    ]
                }
                | {"num_closest_box": PolarsLit(None, dtype=pl.Int64())},
            ),
            create_info_columns=[
                (
                    pl.col("post_box_required").is_null()
                    | pl.col("post_box_required").eq(0)
                ).alias("zero_required"),
                (
                    pl.col("total_post_box_required").is_null()
                    | pl.col("total_post_box_required").eq(0)
                ).alias("zero_total_post_box_required"),
            ],
        )
        .with_columns(channel_struct=pl.struct(Channel.members()))
        .drop(*Channel.members())
        .with_columns(
            channel_struct=(
                pl.when(
                    pl.col.channel_struct.eq(
                        Channel.parse("janandjul.com").as_dict()
                    )
                )
                .then(pl.lit(Channel.parse("jjweb ca east").as_dict()))
                .otherwise(pl.col.channel_struct)
            )
        )
        .unnest("channel_struct"),
    )

    # setup for partitioning information into stuff where the total
    # required for a SKU across all channels is less than what the
    # warehouse has in stock, and the case where the total required is
    # greater than what the channel has in stock
    all_sku_info = all_sku_info.with_columns(
        post_box_required_gt_supply=(
            ~(
                pl.col("total_post_box_required").eq(0)
                | pl.col("total_post_box_required").le(
                    pl.col("wh_dispatchable")
                )
            )
        )
    ).with_columns(
        # for cases where required is not greater than supply, the dispatch
        # is just the required
        dispatch=pl.when(~pl.col("post_box_required_gt_supply"))
        .then(pl.col("post_box_required"))
        .otherwise(None)
    )
    assert all_sku_info["dispatch"].dtype == pl.Int64()

    # filter for the cases where the total required is greater than the
    # supply available
    required_gt_supply = all_sku_info.filter(
        pl.col("post_box_required_gt_supply")
    )

    if required_gt_supply.shape[0] > 0:
        # when total required is greater than supply, partition supply by
        # relative proportion of requested amount
        required_gt_supply = required_gt_supply.with_columns(
            fraction_dispatch=(
                pl.col("pre_box_required") / pl.col("total_pre_box_required")
            )
        ).with_columns(
            post_split_required=(
                pl.col("wh_dispatchable") * pl.col("fraction_dispatch")
            )
            .round()
            .cast(pl.Int64()),
        )

        # redo the full-box shipment logic, this time checking to see the
        # the rounded quantity is strictly less than pre_box_dispatch
        required_gt_supply = (
            required_gt_supply.with_columns(
                post_split_exact_boxes=pl.when(
                    pl.col.enable_full_box_logic
                    & pl.col.qty_box.is_not_null()
                    & pl.col.qty_box.gt(0)
                ).then(pl.col.post_split_required / pl.col.qty_box)
            )
            .with_columns(
                post_split_exact_boxes_floor=pl.col.post_split_exact_boxes.floor().cast(
                    pl.Int64()
                ),
            )
            .with_columns(
                post_split_qty_exact_boxes_floor=(
                    pl.col.qty_box * pl.col.post_split_exact_boxes_floor
                ).cast(pl.Int64),
            )
            .with_columns(
                post_split_distance_to_floor=(
                    pl.col.post_split_required.sub(
                        pl.col.post_split_qty_exact_boxes_floor
                    )
                ),
                post_split_margin_by_ratio=(
                    analysis_defn.full_box_rounding_margin_ratio
                    * pl.col.post_split_required
                )
                .ceil()
                .cast(pl.Int64()),
            )
            .with_columns(
                eb_not_null=pl.col.post_split_exact_boxes.is_not_null(),
                close_by_ratio=pl.col.post_split_distance_to_floor.le(
                    pl.col.post_split_margin_by_ratio
                ),
                close_by_qty=pl.col.post_split_distance_to_floor.le(
                    pl.lit(analysis_defn.full_box_rounding_margin_qty)
                ),
                eb_gt_zero=pl.col.post_split_exact_boxes_floor > 0,
                post_split_close_to_floor=(
                    pl.col.post_split_exact_boxes.is_not_null()
                    & (
                        pl.col.post_split_distance_to_floor.le(
                            pl.col.post_split_margin_by_ratio
                        )
                        | pl.col.post_split_distance_to_floor.le(
                            pl.lit(analysis_defn.full_box_rounding_margin_qty)
                        )
                    )
                    & (pl.col.post_split_exact_boxes_floor > 0)
                ),
            )
            .with_columns(
                post_split_num_closest_box=pl.when(
                    pl.col.post_split_exact_boxes.is_not_null()
                ).then(
                    pl.when(pl.col.post_split_close_to_floor)
                    .then(pl.col.post_split_exact_boxes_floor)
                    .otherwise(pl.lit(None))
                ),
                dispatch=pl.when(
                    pl.col.post_split_exact_boxes.is_not_null()
                ).then(
                    pl.when(pl.col.post_split_close_to_floor)
                    .then(pl.col.post_split_qty_exact_boxes_floor)
                    .otherwise(pl.col.post_split_required)
                ),
            )
        )
        assert required_gt_supply["dispatch"].dtype == pl.Int64()

        # set up to check whether the total dispatch across channels for a
        # SKU is still somehow greater than required (possible due to
        # rounding issues)
        required_gt_supply = required_gt_supply.with_columns(
            total_dispatch=pl.col("dispatch")
            .sum()
            .over(Sku.members(MemberType.META))
        )
        # FINE AUTOSPLIT LOGIC: removed for now after discussion with Matt

        # # partition data set into those which require further fixing, and
        # # those that do not
        # requires_fine_splitting, no_further_fix_required = (
        #     binary_partition_strict(
        #         required_gt_supply,
        #         (pl.col("total_dispatch") > pl.col("wh_dispatchable"))
        #         .and_(pl.col("wh_dispatchable").gt(0))
        #         .and_(pl.col("dispatch").gt(0)),
        #     )
        # )

        # those SKUs which do not require further fixing can now have their
        # dispatch quantity updated in the main info dataframe
        all_sku_info = override_sku_info(
            all_sku_info.with_columns(auto_split=pl.lit(False)),
            # no_futher_fix_required
            required_gt_supply.with_columns(auto_split=pl.lit(True)).select(
                Sku.members(MemberType.META)
                + Channel.members()
                + [
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
                    "dispatch",
                    "auto_split",
                ]
            ),
            fill_null_value=None,
            create_info_columns=None,
            create_missing_info_flags=False,
        )
    else:
        all_sku_info = all_sku_info.with_columns(auto_split=pl.lit(False))

    all_sku_info = all_sku_info.with_columns(
        pl.col("dispatch").fill_null(0),
    )
    assert all_sku_info["dispatch"].dtype == pl.Int64()

    all_sku_info = all_sku_info.with_columns(
        dispatch_below_cutoff=pl.col("dispatch").lt(
            analysis_defn.dispatch_cutoff_qty
        )
    )

    return all_sku_info


class Dispatcher:
    """Manages calculation of a dispatch for FBA refill."""

    analysis_defn: RefillDefn
    """The analysis definition governing this dispatcher."""
    all_sku_info: pl.DataFrame
    """Various information per SKU. To understand it, its best to print it out
    in an interactive environment."""
    filters: FilterStructs
    """Various filters applied to warehouse data source, in order to focus on
    dispatch for particular SKUs and channels."""
    config_data: ConfigData
    """Configuration information from the marketing team."""
    qty_box_info: pl.DataFrame
    """Qty/box information."""
    predictor: Predictor
    """Used to predict the demand for the prediction period."""
    dispatch_start: Date
    """Start of the prediction period."""
    dispatch_end: Date
    """End of the prediction period."""
    channel_info: pl.DataFrame
    """Channel information dataframe."""
    dispatch_channels: list[Channel]
    """Channels to focus on for FBA refill."""
    current_year: int
    """Current year (relative to the prediction start date)."""
    next_year: int
    """Next year (relative to the prediction start date)."""
    reserved_quantity: pl.DataFrame | None
    """Reserved quantity dataframe."""
    expected_demand_last_year: pl.DataFrame | None
    """Expected demand if current period is taken to be the last year."""

    @property
    def db(self) -> DataBase:
        """The database backing this prediction."""
        return self.predictor.db

    def __init__(
        self,
        analysis_defn_or_db: RefillDefn | DataBase,
        predictor: Predictor,
        filters: list[StructLike] | None = [],
        read_from_disk: bool = False,
        overwrite: bool = True,
        dispatch_start: DateLike | None = None,
        dispatch_end: DateLike | None = None,
    ) -> None:
        analysis_defn, db = get_analysis_defn_and_db(analysis_defn_or_db)
        assert isinstance(analysis_defn, RefillDefn)

        self.analysis_defn = analysis_defn
        # read marketing configuration info
        # channel specific modifications to the config data (e.g. setting refill
        # request to 1 unit for JanAndJul.com website) are handled when this
        # data is attached to the output dataframe
        # (see attach_refill_info_from_config)
        # (see attach_inventory_info)

        # read qty/box information
        self.qty_box_info = read_qty_box(
            analysis_defn, read_from_disk=read_from_disk, overwrite=overwrite
        )
        # parse the given focus channels as Channels
        self.dispatch_channels = analysis_defn.channels
        self.predictor = predictor
        self.filters = FilterStructs(
            list(
                (
                    set(self.dispatch_channels)
                    | set([Channel.parse(x) for x in ["Warehouse CA"]])
                    | set(normalize_optional(filters, []))
                )
            )
        )

        # get channel information for the channels we want to dispatch to
        self.channel_info = struct_filter(
            self.db.meta_info.channel.select(Channel.members()),
            set(self.dispatch_channels),
        ).unique()

        self.config_data = (
            read_config(analysis_defn)
            .extra_refill_info(
                db.meta_info.active_sku, analysis_defn.extra_refill_config_info
            )
            .filter_channels(self.channel_info)
        )

        # initialize all sku information, which we will further build upon as
        # the dispatch calculations progress
        self.all_sku_info = db.meta_info.all_sku.select(
            "status",
            *(
                ALL_SKU_IDS
                + SEASON_IDS
                + [
                    x
                    for x in PAUSE_PLAN_IDS + STATUS_IDS + NOVELTY_FLAGS
                    if x in db.meta_info.active_sku.columns
                ]
            ),
        ).with_columns(is_active=pl.col.status.eq("active"))

        if "website_sku" not in self.all_sku_info:
            self.all_sku_info = self.all_sku_info.with_columns(
                website_sku=pl.lit(False)
            )

        self.all_sku_info = attach_channel_info(
            self.all_sku_info, self.channel_info
        )

        # do a duplicate check, in order to catch any duplicates (there should
        # be no duplicate entries for each SKU + channel)
        find_dupes(
            self.all_sku_info, ALL_SKU_AND_CHANNEL_IDS, raise_error=True
        )
        assert len(
            self.all_sku_info.select(Channel.members()).unique()
        ) == len(self.dispatch_channels)

        # select particular columns out of all_sku_info
        self.all_sku_info = self.all_sku_info.select(
            ALL_IDS + PAUSE_PLAN_IDS + MASTER_PAUSE_FLAGS + NOVELTY_FLAGS
        )

        self.all_sku_info = attach_refill_info_from_config(
            self.all_sku_info, self.config_data
        )
        # do a duplicate check, in order to catch any duplicates (there should
        # be no duplicate entries for each SKU + channel)
        find_dupes(
            self.all_sku_info, ALL_SKU_AND_CHANNEL_IDS, raise_error=True
        )
        assert len(
            self.all_sku_info.select(Channel.members()).unique()
        ) == len(self.dispatch_channels)

        if self.analysis_defn.jjweb_reserve_info is None:
            self.reserved_quantity = None
        else:
            self.reserved_quantity = calculate_reserved_quantity(
                db,
                self.predictor,
                self.analysis_defn.jjweb_reserve_info.reservation_expr,
                force_po_predictions=self.analysis_defn.jjweb_reserve_info.force_reserve_po_prediction,
            )

        if (
            self.reserved_quantity is not None
            and len(self.reserved_quantity) > 0
        ):
            self.all_sku_info = self.all_sku_info.join(
                self.reserved_quantity.group_by("sku", "a_sku")
                .agg(pl.col.reserved.sum().round().cast(pl.Int64()))
                .select(
                    "sku",
                    "a_sku",
                    pl.col.reserved,
                ),
                on=["sku", "a_sku"],
                how="left",
            ).with_columns(pl.col.reserved.fill_null(0))
        else:
            self.all_sku_info = self.all_sku_info.with_columns(
                reserved=pl.lit(0)
            )

        if Channel.parse("jjweb ca east") in self.dispatch_channels:
            input_data_info = self.predictor.get_input_data_info().join(
                self.channel_info, on=Channel.members()
            )

            self.expected_demand_last_year = (
                calculate_jjweb_past_one_year_quantity(db, predictor).join(
                    input_data_info.select(
                        "sku",
                        *Channel.members(),
                        "prediction_type",
                        "has_po_data",
                    ).filter(
                        pl.col.prediction_type.eq(PredictionType.PO.name)
                        & ~pl.col.has_po_data
                    ),
                    on=["sku", *Channel.members()],
                    validate="1:1",
                )
            )

            if len(self.expected_demand_last_year) > 0:
                self.all_sku_info = self.all_sku_info.join(
                    self.expected_demand_last_year.select(
                        "sku",
                        *Channel.members(),
                        pl.col.expected_demand.alias(
                            "expected_demand_last_year"
                        ),
                    ),
                    on=["sku"] + Channel.members(),
                    how="left",
                )
        else:
            self.expected_demand_last_year = pl.DataFrame()
            self.all_sku_info = self.all_sku_info.with_columns(
                expected_demand_last_year=pl.lit(0)
            )

        # attach warehouse stock information and min_keep_qty information
        self.all_sku_info = attach_inventory_info(
            self.db,
            self.all_sku_info,
            pl.col("platform")
            .eq("Warehouse")
            .and_(pl.col("mode").eq(DistributionMode.WAREHOUSE.name)),
            analysis_defn.warehouse_min_keep_qty,
            self.filters,
        ).with_columns(
            reserved_before_on_order=pl.col.reserved,
            reserved=pl.max_horizontal(
                pl.col.reserved - pl.col.wh_on_order, 0
            ),
        )
        assert len(
            self.all_sku_info.select(Channel.members()).unique()
        ) == len(self.dispatch_channels)
        if "ch_stock" not in self.all_sku_info.columns:
            self.all_sku_info = self.all_sku_info.with_columns(
                ch_stock=pl.when(pl.col.is_active).then(pl.lit(0))
            )

        # jjweb_channel = Channel.parse("janandjul.com")

        skus_with_reservation = self.all_sku_info.filter(
            pl.col.reserved.gt(0)
        )["sku"].unique()

        # TODO: need to get the JJWEB EAST 3PL inventory from sales/channel data
        self.all_sku_info = self.all_sku_info.with_columns(
            jjweb_inv_3pl=pl.lit(0)
        )
        # TODO: need to calculate the JJWEB EAST fractions based on sales data
        self.all_sku_info = self.all_sku_info.with_columns(
            jjweb_east_frac=pl.lit(0.0)
        )

        self.all_sku_info = (
            self.all_sku_info.with_columns(
                has_reservation=pl.col.sku.is_in(skus_with_reservation)
            )
            .with_columns(
                reserved=pl.when(
                    pl.col.platform.eq(Platform.JJWeb.name)
                    # pl.struct(Channel.members()).eq(jjweb_channel.as_dict())
                )
                .then(pl.lit(0))
                .otherwise(pl.col.reserved)
            )
            .with_columns(
                reserved_west=((1 - pl.col.jjweb_east_frac) * pl.col.reserved)
                .ceil()
                .cast(pl.Int64()),
                reserved_including_3pl=pl.max_horizontal(
                    0, pl.col.reserved - pl.col.jjweb_inv_3pl
                ),
            )
            .with_columns(
                wh_dispatchable_accounting_jjweb_west=(
                    pl.col.wh_stock - pl.col.min_keep - pl.col.reserved_west
                ),
                wh_dispatchable_accounting_jjweb_east=(
                    pl.col.wh_stock
                    - pl.col.min_keep
                    - pl.col.reserved_including_3pl
                ),
            )
            .with_columns(
                pl.when(pl.col("wh_stock").gt(0))
                .then(
                    pl.max_horizontal(
                        pl.lit(0),
                        pl.min_horizontal(
                            pl.col.wh_dispatchable_accounting_jjweb_west,
                            pl.col.wh_dispatchable_accounting_jjweb_east,
                        ),
                    ).alias("wh_dispatchable")
                )
                .otherwise(
                    pl.lit(0, dtype=pl.Int64()).alias("wh_dispatchable")
                )
            )
            .with_columns(
                (
                    pl.col("wh_stock").is_null()
                    | pl.col("wh_dispatchable").eq(0)
                ).alias("zero_wh_dispatchable")
            )
        )
        find_dupes(
            self.all_sku_info, ALL_SKU_AND_CHANNEL_IDS, raise_error=True
        )
        assert len(
            self.all_sku_info.select(Channel.members()).unique()
        ) == len(self.dispatch_channels)

        # attach input data information
        # self.all_sku_info = override_sku_info(
        #     self.all_sku_info,
        #     input_data_info,
        #     fill_null_value=None,
        #     create_info_columns=[
        #         "current_period_sales",
        #     ],
        #     create_missing_info_flags=False,
        #     nulls_equal=False,
        #     dupe_check_index=ALL_SKU_AND_CHANNEL_IDS,
        # )
        # find_dupes(
        #     self.all_sku_info, ALL_SKU_AND_CHANNEL_IDS, raise_error=True
        # )
        # assert len(
        #     self.all_sku_info.select(Channel.members()).unique()
        # ) == len(self.dispatch_channels)

        self.dispatch_start = Date.from_datelike(
            dispatch_start if dispatch_start else analysis_defn.dispatch_date
        )
        self.dispatch_end = Date.from_datelike(
            dispatch_end if dispatch_end else analysis_defn.end_date
        )

        self.all_sku_info = self.all_sku_info.with_columns(
            dispatch_start=self.dispatch_start.as_polars_date(),
            dispatch_end=self.dispatch_end.as_polars_date(),
        )

        # attach demand predictions
        demand_predictions = self.predictor.predict_demand(
            self.dispatch_channels,
            self.dispatch_start,
            self.dispatch_end,
        ).filter(
            ~(
                # pl.struct(Channel.members()).eq(jjweb_channel.as_dict())
                pl.col.platform.eq(Platform.JJWeb.name)
                & pl.col.sku.is_in(skus_with_reservation)
            )
        )

        if self.analysis_defn.jjweb_reserve_info is not None:
            if any(
                (x.platform == Platform.JJWeb.name)
                for x in self.dispatch_channels
            ):
                # if jjweb_channel in self.dispatch_channels:
                demand_predictions = concat_enum_extend_vstack_strict(
                    [
                        demand_predictions,
                        self.predictor.predict_demand(
                            [
                                x
                                for x in self.dispatch_channels
                                if x.platform == Platform.JJWeb.name
                                # if x == jjweb_channel
                            ],
                            self.analysis_defn.dispatch_date,
                            self.analysis_defn.jjweb_reserve_info.prediction_offset_3pl_when_reservation_on.apply_to(
                                analysis_defn.dispatch_date
                            ),
                        ).filter(pl.col.sku.is_in(skus_with_reservation)),
                    ]
                )

        self.all_sku_info = self.all_sku_info.join(
            self.config_data.in_config_file, on=["category"], how="left"
        ).with_columns(pl.col.in_config_file.fill_null(pl.lit(False)))

        find_dupes(
            demand_predictions,
            ["sku", "a_sku"] + Channel.members(),
            raise_error=True,
        )

        self.all_sku_info = (
            override_sku_info(
                self.all_sku_info,
                demand_predictions.select(
                    ["sku", "a_sku"]
                    + Channel.members()
                    + [
                        x
                        for x in demand_predictions.columns
                        if x not in self.all_sku_info.columns
                    ]
                ),
                fill_null_value=defaultdict(
                    lambda: None,
                    {
                        "e_overrides_po": PolarsLit(0),
                        "po_overrides_e": PolarsLit(0),
                    },
                ),
                create_info_columns=[
                    "expected_demand",
                    "expected_demand_from_history",
                    "expected_demand_from_po",
                ],
            )
            .with_columns(
                applies_missing_po_consideration=(
                    (pl.col.uses_po & ~pl.col.has_po_data & ~pl.col.uses_ne)
                    & pl.struct(Channel.members()).eq(
                        Channel.parse("jjweb ca east").as_dict()
                    )
                )
            )
            .with_columns(
                expected_demand_before_missing_po_consideration=pl.when(
                    pl.col.applies_missing_po_consideration
                )
                .then(pl.col.refill_request)
                .otherwise(pl.col.expected_demand),
            )
            .with_columns(
                expected_demand=pl.when(
                    pl.col.applies_missing_po_consideration
                )
                .then(pl.col.expected_demand_last_year)
                .otherwise(
                    pl.col.expected_demand_before_missing_po_consideration
                )
            )
        )
        find_dupes(
            self.all_sku_info, ALL_SKU_AND_CHANNEL_IDS, raise_error=True
        )
        assert len(
            self.all_sku_info.select(Channel.members()).unique()
        ) == len(self.dispatch_channels)

        # attach qty/box information
        self.all_sku_info = override_sku_info(
            self.all_sku_info,
            self.qty_box_info,
            create_missing_info_flags=True,
            dupe_check_index=ALL_SKU_AND_CHANNEL_IDS,
        ).with_columns(
            no_qty_box_info=pl.when(pl.col.qty_box.eq(0))
            .then(pl.lit(True))
            .otherwise(pl.col.no_qty_box_info)
        )

        find_dupes(
            self.all_sku_info, ALL_SKU_AND_CHANNEL_IDS, raise_error=True
        )
        assert len(
            self.all_sku_info.select(Channel.members()).unique()
        ) == len(self.dispatch_channels)

    def calculate_dispatch(
        self,
        overwrite: bool = True,
    ) -> pl.DataFrame:
        reserve_on, reserve_off = binary_partition_strict(
            self.all_sku_info, pl.col.has_reservation
        )

        reserve_on_jjweb, reserve_on_others = binary_partition_strict(
            reserve_on,
            pl.struct(Channel.members()).eq(
                Channel.parse("janandjul.com").as_dict()
            ),
        )

        self.all_sku_info = concat_enum_extend_vstack_strict(
            [
                calculate_full_box_and_auto_split(self.analysis_defn, x)
                for x in [reserve_off, reserve_on_jjweb, reserve_on_others]
                if len(x) > 0
            ]
        )

        find_dupes(
            self.all_sku_info, ALL_SKU_AND_CHANNEL_IDS, raise_error=True
        )

        write_df(
            overwrite,
            ANALYSIS_OUTPUT_FOLDER.joinpath(
                Path(
                    f"calculated_dispatch_{self.analysis_defn.tag_with_output_time()}.parquet"
                )
            ),
            self.all_sku_info,
        )

        return self.all_sku_info

    def calculate_dispatch_special_case(
        self, required_ca: pl.DataFrame
    ) -> pl.DataFrame:
        assert len(self.all_sku_info.filter(pl.col.reserved.ne(0))) == 0

        us_stuff, other_stuff = binary_partition_strict(
            self.all_sku_info,
            pl.struct(Channel.members()).eq(
                Channel.parse("Amazon US").as_dict()
            ),
        )

        assert len(other_stuff) == 0

        # assert len(us_stuff) + len(ca_stuff) == len(self.all_sku_info)

        # assert (
        #     us_stuff["sku"].unique().sort() == ca_stuff["sku"].unique().sort()
        # ).all()

        us_stuff = (
            us_stuff.join(
                required_ca,
                on=["sku"],
                how="left",
            )
            .with_columns(pl.col.required_CA.fill_null(0))
            .with_columns(
                wh_dispatchable=pl.max_horizontal(
                    pl.col.wh_dispatchable - pl.col.required_CA, pl.lit(0)
                )
            )
        )

        return calculate_full_box_and_auto_split(self.analysis_defn, us_stuff)

    def produce_formatted_jjweb_dispatch(
        self,
        calculated_dispatch: pl.DataFrame | None = None,
        save_excel: bool = True,
        dispatch_filter: pl.Expr | None = None,
        descriptor: str | None = None,
        save_csv: bool = True,
    ) -> pl.DataFrame:
        """Prepare an Excel file in the expected dispatch output format based on
        calculated dispatch.

        If a calculated dispatch is not given, then one is calculated from
        scratch."""

        if calculated_dispatch is None:
            calculated_dispatch = self.calculate_dispatch()

        final_dispatch = calculated_dispatch.filter(pl.col.dispatch.gt(0))

        if dispatch_filter is not None:
            final_dispatch = final_dispatch.filter(dispatch_filter)

        # raise NotImplementedError("Not yet implemented!")

        sheets = {
            "EAST": format_jjweb_dispatch_for_netsuite(
                self.analysis_defn.date,
                final_dispatch,
                CountryFlags.CA,
            ),
        }

        if descriptor is not None:
            descriptor = f"_{descriptor}"
        else:
            descriptor = ""

        if save_excel:
            result_path = ANALYSIS_OUTPUT_FOLDER.joinpath(
                Path(
                    f"{self.analysis_defn.tag()}_final_dispatch{descriptor}.xlsx"
                )
            )
            write_excel(result_path, sheets)
        if save_csv:
            for x in sheets.keys():
                result_path = ANALYSIS_OUTPUT_FOLDER.joinpath(
                    Path(
                        f"TO_SRR_JJWEB{x}{self.analysis_defn.date.fmt_flat()}.csv"
                    )
                )
                sheets[x].write_csv(result_path)

        return final_dispatch

    def produce_formatted_fba_dispatch(
        self,
        calculated_dispatch: pl.DataFrame | None = None,
        save_excel: bool = True,
        dispatch_filter: pl.Expr | None = None,
        descriptor: str | None = None,
        save_csv: bool = True,
        extra_cols: list[str] = [],
    ) -> pl.DataFrame:
        """Prepare an Excel file in the expected dispatch output format based on
        calculated dispatch.

        If a calculated dispatch is not given, then one is calculated from
        scratch."""

        if calculated_dispatch is None:
            calculated_dispatch = self.calculate_dispatch()

        final_dispatch = calculated_dispatch.filter(
            pl.col("is_active")
            & ~(
                pl.col("is_master_paused")
                | pl.col("is_config_paused")
                | pl.col("dispatch_below_cutoff")
            )
        ).join(
            self.db.meta_info.fba_sku,
            on=["sku", "country_flag"],
            how="left",
            validate="1:1",
        )

        if dispatch_filter is not None:
            final_dispatch = final_dispatch.filter(dispatch_filter)

        sheets = {
            "US": format_fba_dispatch_for_netsuite(
                self.analysis_defn.date,
                final_dispatch,
                CountryFlags.US,
                extra_cols=extra_cols,
            ),
            "CA": format_fba_dispatch_for_netsuite(
                self.analysis_defn.date,
                final_dispatch,
                CountryFlags.CA,
                extra_cols=extra_cols,
            ),
        }

        if descriptor is not None:
            descriptor = f"_{descriptor}"
        else:
            descriptor = ""

        if save_excel:
            result_path = ANALYSIS_OUTPUT_FOLDER.joinpath(
                Path(
                    f"{self.analysis_defn.tag()}_final_dispatch{descriptor}.xlsx"
                )
            )
            write_excel(result_path, sheets)
        if save_csv:
            if self.analysis_defn.extra_descriptor != "":
                extra_descriptor = f"-{self.analysis_defn.extra_descriptor}"
            else:
                extra_descriptor = ""

            if descriptor != "":
                extra_descriptor += descriptor

            for x in sheets.keys():
                result_path = ANALYSIS_OUTPUT_FOLDER.joinpath(
                    Path(
                        f"TO_SRR_FBA{x}{self.analysis_defn.date.fmt_flat()}{extra_descriptor}.csv"
                    )
                )
                sheets[x].write_csv(result_path)

        return final_dispatch
