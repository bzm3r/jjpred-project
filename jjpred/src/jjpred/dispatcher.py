"""Logic for calculating a dispatch for FBA refill."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import polars as pl

from jjpred.analysisdefn import FbaRevDefn
from jjpred.channel import Channel, DistributionMode
from jjpred.datagroups import (
    ALL_IDS,
    ALL_SKU_AND_CHANNEL_IDS,
    MASTER_PAUSE_FLAGS,
    PAUSE_PLAN_IDS,
    NOVELTY_FLAGS,
)
from jjpred.countryflags import CountryFlags
from jjpred.database import DataBase
from jjpred.dispatchformatter import format_dispatch_for_netsuite
from jjpred.globalpaths import ANALYSIS_OUTPUT_FOLDER
from jjpred.globalvariables import (
    DISPATCH_CUTOFF_QTY,
)
from jjpred.predictor import Predictor
from jjpred.readsupport.qtybox import read_qty_box
from jjpred.readsupport.marketing import ConfigData, read_config
from jjpred.scripting.dateoffset import (
    determine_main_program_compatible_start_end_dates,
)
from jjpred.sku import Sku
from jjpred.skuinfo import (
    attach_inventory_info,
    override_sku_info,
    attach_channel_info,
    attach_refill_info_from_config,
    get_all_sku_currentness_info,
)
from jjpred.structlike import MemberType, StructLike
from jjpred.utils.datetime import Date
from jjpred.utils.fileio import write_df, write_excel
from jjpred.utils.polars import (
    FilterStructs,
    find_dupes,
    struct_filter,
)
from jjpred.utils.typ import PolarsLit, normalize_optional


def calculate_required(
    predicted_demand: pl.DataFrame,
    enable_full_box_logic: bool,
) -> pl.DataFrame:
    """Calculate the quantity of goods required by a channel, given its
    predicted demand.

    This quantity depends on:
    * the current inventory of the channel (we subtract the current inventory
    from the predicted demand quantity to get the ``prebox_required``
    (pre-full-box logic required amount)
    * we round ``prebox_required/qty_box`` to get how many boxes of goods
    are required (``exact_boxes``)

      * if ``prebox_required`` is within 10% of this quantity, then we
        use ``exact_boxes * qty_box`` as the ``required`` amount for the
        channel
      * otherwise we ``prebox_required`` as the ``required`` amount
    """
    df = predicted_demand

    assert df["expected_demand"].dtype == pl.Int64()
    df = df.with_columns(
        requesting=pl.max_horizontal(
            pl.col("expected_demand").ceil().cast(pl.Int64()),
            pl.col("refill_request"),
        )
    ).with_columns(
        uses_refill_request=(
            pl.col("requesting").eq(pl.col("refill_request"))
            & pl.col.refill_request.gt(0)
        )
    )

    df = df.with_columns(
        prebox_required=pl.when(pl.col("requesting") > pl.col("ch_stock"))
        .then(pl.col("requesting") - pl.col("ch_stock"))
        .otherwise(0)
    )

    df = (
        df.with_columns(
            enable_full_box_logic=pl.lit(
                enable_full_box_logic, dtype=pl.Boolean()
            )
        )
        .with_columns(
            exact_boxes=pl.when(
                pl.col.qty_box.is_not_null() & pl.col.qty_box.gt(0)
            )
            .then((pl.col("prebox_required") / pl.col("qty_box")).round())
            .cast(pl.Int64())
        )
        .with_columns(
            rounded_to_closest_box=pl.when(
                pl.col.exact_boxes.is_not_null()
                & pl.col.prebox_required.gt(0)
                & pl.col.enable_full_box_logic
            )
            .then(
                (0.9 * pl.col.qty_box * pl.col.exact_boxes).le(
                    pl.col.prebox_required
                )
                & (pl.col.prebox_required).le(
                    1.1 * pl.col.qty_box * pl.col.exact_boxes
                )
            )
            .otherwise(False)
        )
        .with_columns(
            required=pl.when(pl.col.rounded_to_closest_box)
            .then(pl.col.exact_boxes * pl.col.qty_box)
            .otherwise(pl.col.prebox_required)
        )
    )

    df = df.with_columns(
        total_required=pl.col("required")
        .sum()
        .over(
            [
                x
                for x in df.columns
                if (x in ["a_sku"] + Sku.members(MemberType.META))
            ]
        )
    )

    assert df["required"].dtype == pl.Int64()

    return df


class Dispatcher:
    """Manages calculation of a dispatch for FBA refill."""

    analysis_defn: FbaRevDefn
    """The analysis definition governing this dispatcher."""
    all_sku_info: pl.DataFrame
    """Various information per SKU. To understand it, its best to print it out
    in an interactive environment."""
    filters: FilterStructs
    """Various filters applied to warehouse data source, in order to focus on
    dispatch for particular SKUs and channels."""
    dispatch_cutoff: int
    """The minimum dispatch size needed in order for an SKU to be included in a
    final dispatch."""
    config_data: ConfigData
    """Configuration information from the marketing team."""
    qty_box_info: pl.DataFrame
    """Qty/box information."""
    predictor: Predictor
    """Used to predict the demand for the prediction period."""
    start_date: Date
    """Start of the prediction period."""
    end_date: Date
    """End of the prediction period."""
    channel_info: pl.DataFrame
    """Channel information dataframe."""
    dispatch_channels: list[Channel]
    """Channels to focus on for FBA refill."""
    current_year: int
    """Current year (relative to the prediction start date)."""
    next_year: int
    """Next year (relative to the prediction start date)."""

    @property
    def db(self) -> DataBase:
        """The database backing this prediction."""
        return self.predictor.db

    def __init__(
        self,
        analysis_defn: FbaRevDefn,
        dispatch_channels: list[Channel | str],
        predictor: Predictor,
        dispatch_cutoff: int = DISPATCH_CUTOFF_QTY,
        filters: list[StructLike] | None = [],
        read_from_disk: bool = False,
    ) -> None:
        self.analysis_defn = analysis_defn
        # read marketing configuration info
        self.config_data = read_config(analysis_defn)
        # read qty/box information
        self.qty_box_info = read_qty_box(
            analysis_defn, read_from_disk=read_from_disk
        )
        # parse the given focus channels as Channels
        self.dispatch_channels = list(
            Channel.parse(x) for x in dispatch_channels
        )
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
        self.dispatch_cutoff = dispatch_cutoff

        # get channel information the channels we want to dispatch to
        self.channel_info = struct_filter(
            self.db.meta_info.channel.select(Channel.members()),
            set(self.dispatch_channels),
        ).unique()
        # initialize all sku information, which we will further build upon as
        # the dispatch calculations progress
        self.all_sku_info = get_all_sku_currentness_info(analysis_defn)

        self.all_sku_info = attach_channel_info(
            self.all_sku_info, self.channel_info
        )
        # do a duplicate check, in order to catch any duplicates (there should
        # be no duplicate entries for each SKU + channel)
        find_dupes(
            self.all_sku_info, ALL_SKU_AND_CHANNEL_IDS, raise_error=True
        )

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

        # attach warehouse stock information and min_keep_qty information
        self.all_sku_info = attach_inventory_info(
            self.db,
            self.config_data,
            self.all_sku_info,
            pl.col("platform")
            .eq("Warehouse")
            .and_(pl.col("mode").eq(DistributionMode.WAREHOUSE.name)),
            self.filters,
        )
        find_dupes(
            self.all_sku_info, ALL_SKU_AND_CHANNEL_IDS, raise_error=True
        )

        # determine start/end dates that are compatible with the main program
        # TODO: this should be changed to just use the actual start/end dates,
        # once we are no longer comparing with the main program
        self.start_date, self.end_date = (
            determine_main_program_compatible_start_end_dates(
                analysis_defn.dispatch_date,
                analysis_defn.refill_type,
                start_date_required_month_parts=analysis_defn.prediction_start_date_required_month_parts,
                end_date_required_month_parts=analysis_defn.prediction_end_date_required_month_parts,
            )
        )

        input_data_info = self.predictor.get_input_data_info()
        # attach input data information
        self.all_sku_info = override_sku_info(
            self.all_sku_info,
            input_data_info,
            fill_null_value=None,
            create_info_columns=[
                "current_period_sales",
            ],
            create_missing_info_flags=False,
            join_nulls=False,
            dupe_check_index=ALL_SKU_AND_CHANNEL_IDS,
        )

        find_dupes(
            self.all_sku_info, ALL_SKU_AND_CHANNEL_IDS, raise_error=True
        )

        # attach demand predictions
        demand_predictions = self.predictor.predict_demand(
            self.dispatch_channels, self.start_date, self.end_date
        )

        self.all_sku_info = (
            override_sku_info(
                self.all_sku_info,
                demand_predictions,
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
                refill_request_override=(
                    (pl.col.prediction_type.eq("E") & ~pl.col.has_e_data)
                    | (
                        pl.col.prediction_type.eq("PO")
                        & ~pl.col.uses_ne
                        & ~pl.col.has_po_data
                        & ~pl.col.e_overrides_po
                    )
                    | (pl.col.uses_ne & pl.col.new_category_problem)
                )
                & pl.col("refill_request").is_not_null()
                & pl.col("refill_request").gt(0)
            )
            .with_columns(
                expected_demand=pl.when(
                    pl.col("refill_request_override").gt(0)
                )
                .then(pl.col("refill_request"))
                .otherwise(pl.col("expected_demand")),
            )
        )
        find_dupes(
            self.all_sku_info, ALL_SKU_AND_CHANNEL_IDS, raise_error=True
        )

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

    def calculate_dispatch(
        self,
        overwrite: bool = True,
    ) -> pl.DataFrame:
        # calculated what each channel + SKU requires for dispatch
        required_df = calculate_required(
            self.all_sku_info.filter(
                pl.col("is_active")
                & ~pl.col("is_master_paused")
                & ~pl.col("is_config_paused")
                & ~pl.col("no_wh_stock_info")
                & ~pl.col("zero_wh_dispatchable")
            ),
            self.analysis_defn.enable_full_box_logic,
        )
        self.all_sku_info = override_sku_info(
            self.all_sku_info,
            required_df,
            fill_null_value=defaultdict(
                lambda: PolarsLit(0),
                {
                    "uses_refill_request": PolarsLit(False),
                    "enable_full_box_logic": PolarsLit(False),
                    "rounded_to_closest_box": PolarsLit(False),
                },
            ),
            create_info_columns=[
                (
                    pl.col("required").is_null() | pl.col("required").eq(0)
                ).alias("zero_required"),
                (
                    pl.col("total_required").is_null()
                    | pl.col("total_required").eq(0)
                ).alias("zero_total_required"),
            ],
        )

        # setup for partitioning information into stuff where the total
        # required for a SKU across all channels is less than what the
        # warehouse has in stock, and the case where the total required is
        # greater than what the channel has in stock
        self.all_sku_info = self.all_sku_info.with_columns(
            required_gt_supply=(
                ~(
                    pl.col("total_required").eq(0)
                    | pl.col("wh_dispatchable").eq(0)
                    | pl.col("total_required").le(pl.col("wh_dispatchable"))
                )
            )
        ).with_columns(
            # for cases where required is not greater than supply, the dispatch
            # is just the required
            dispatch=pl.when(~pl.col("required_gt_supply"))
            .then(pl.col("required"))
            .otherwise(None)
        )
        assert self.all_sku_info["dispatch"].dtype == pl.Int64()

        # filter for the cases where the total required is greater than the
        # supply available
        required_gt_supply = self.all_sku_info.filter(
            pl.col("required_gt_supply")
        )

        if required_gt_supply.shape[0] > 0:
            # when total required is greater than supply, partition supply by
            # relative proportion of requested amount
            required_gt_supply = required_gt_supply.with_columns(
                fraction_dispatch=(
                    pl.col("required") / pl.col("total_required")
                )
            ).with_columns(
                prebox_required=(
                    pl.col("wh_dispatchable") * pl.col("fraction_dispatch")
                )
                .round()
                .cast(pl.Int64()),
            )

            # redo the full-box shipment logic, this time checking to see the
            # the rounded quantity is strictly less than prebox_dispatch
            required_gt_supply = (
                required_gt_supply.with_columns(
                    exact_boxes=pl.when(
                        pl.col.qty_box.is_not_null() & pl.col.qty_box.gt(0)
                    )
                    .then(
                        (pl.col("prebox_required") / pl.col("qty_box")).round()
                    )
                    .cast(pl.Int64())
                )
                .with_columns(
                    rounded_to_closest_box=pl.when(
                        pl.col.exact_boxes.is_not_null()
                        & pl.col.prebox_required.gt(0)
                        & pl.col.enable_full_box_logic
                    )
                    .then(
                        (1.0 * pl.col.qty_box * pl.col.exact_boxes).le(
                            pl.col.prebox_required
                        )
                        # & (0.9 * pl.col.qty_box * pl.col.exact_boxes).le(
                        #     pl.col.prebox_dispatch
                        # )
                        & (pl.col.prebox_required).le(
                            1.1 * pl.col.qty_box * pl.col.exact_boxes
                        )
                    )
                    .otherwise(False)
                )
                .with_columns(
                    dispatch=pl.when(pl.col.rounded_to_closest_box)
                    .then(pl.col.exact_boxes * pl.col.qty_box)
                    .otherwise(pl.col.prebox_required)
                )
            )
            assert required_gt_supply["required"].dtype == pl.Int64()

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
            self.all_sku_info = override_sku_info(
                self.all_sku_info.with_columns(auto_split=pl.lit(False)),
                # no_futher_fix_required
                required_gt_supply.with_columns(
                    auto_split=pl.lit(True)
                ).select(
                    Sku.members(MemberType.META)
                    + Channel.members()
                    + [
                        "prebox_required",
                        "dispatch",
                        "rounded_to_closest_box",
                        "auto_split",
                    ]
                ),
                fill_null_value=None,
                create_info_columns=None,
                create_missing_info_flags=False,
            )

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
            #                 "prebox_required",
            #                 "dispatch",
            #                 "rounded_to_closest_box",
            #                 "auto_split",
            #                 "fine_auto_split",
            #             ]
            #         ),
            #         fill_null_value=None,
            #         create_info_columns=None,
            #         create_missing_info_flags=False,
            #     ).with_columns(pl.col.auto_split.fill_null(False))

        self.all_sku_info = self.all_sku_info.with_columns(
            pl.col("dispatch").fill_null(0),
        )
        assert self.all_sku_info["dispatch"].dtype == pl.Int64()

        self.all_sku_info = self.all_sku_info.with_columns(
            dispatch_below_cutoff=pl.col("dispatch").lt(self.dispatch_cutoff)
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

    def produce_formatted_dispatch(
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
            "US": format_dispatch_for_netsuite(
                self.analysis_defn.date,
                final_dispatch,
                CountryFlags.US,
            ),
            "CA": format_dispatch_for_netsuite(
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
                        f"TO_SRR_FBA{x}{self.analysis_defn.date.fmt_flat()}.csv"
                    )
                )
                sheets[x].write_csv(result_path)
            # print(f"Saving final dispatch to: {result_path}")

            # if result_path.exists():
            #     result_path.unlink()
            # with xlw.Workbook(result_path) as workbook:
            #     for key, df in sheets.items():
            #         if df is not None:
            #             df.write_excel(workbook=workbook, worksheet=key)

        return final_dispatch
