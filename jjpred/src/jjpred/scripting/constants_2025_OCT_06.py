"""Information used  in order to set up and/or execute analyses."""

from __future__ import annotations

from jjpred.analysisdefn import (
    DEFAULT_RESERVATION_EXPR,
    CurrentSeasonDefn,
    FbaRevDefn,
    FbaRevDefnArgs,
    JJWebPredictionInfo,
)
from jjpred.inputstrategy import RefillType

from jjpred.analysisconfig import GeneralRefillConfigInfo

import polars as pl

args = FbaRevDefnArgs(
    analysis_date="2025-OCT-06",
    current_seasons=CurrentSeasonDefn(FW=25, SS=25),
    dispatch_date="2025-OCT-06",
    master_sku_date="2025-OCT-06",
    sales_and_inventory_date="2025-OCT-06",
    warehouse_inventory_date="2025-OCT-06",
    in_stock_ratio_date="2025-OCT-06",
    website_sku_date="2025-SEP-18",
    config_date="2025-SEP-29",
    prediction_type_meta_date=None,
    check_dispatch_date=False,
    mainprogram_date=None,  # "2025-FEB-25",
    refill_draft_date=None,  # "2025-FEB-25",
    match_main_program_month_fractions=False,
    extra_refill_config_info=(
        [
            GeneralRefillConfigInfo(
                ["amazon.ca", "amazon.com"], 10, pl.col.category.eq("BSL")
            ),
            GeneralRefillConfigInfo(
                ["amazon.ca", "amazon.com"],
                10,
                pl.col.category.eq("SMF") & pl.col.print.is_in(["DPK", "SBR"]),
            ),
            GeneralRefillConfigInfo(
                ["amazon.ca", "amazon.com"],
                5,
                pl.col.min_refill_request.lt(5) & pl.col.season.ne("SS"),
            ),
        ]
    ),
    combine_hca0_hcb0_gra_asg_history=True,
    refill_type=RefillType.END_OF_DEC_2025,
    mon_sale_r_date=None,
    po_date=None,
    new_overrides_e=True,
    forced_po_categories=["IHT"],
    new_categories=[
        "AJA",
        "AJC",
        "BSL",
        "BST",
        "DRC",
        "FAN",
        "FHA",
        "FVM",
        "IPC",
        "IPS",
        "ISB",
        "ISJ",
        "ISS",
        "LAB",
        "LAN",
        "LBP",
        "LBT",
        # "SKX",
        "SBS",
        "SMF",
        "SSW",
        "SWS",
        "WGS",
        "WRM",
        "XBK",
        "XBM",
        "XLB",
        "XPC",
        "XWG",
    ],
    enable_full_box_logic=True,
    full_box_rounding_margin_qty=10,
    full_box_rounding_margin_ratio=0.2,
)

analysis_defn = FbaRevDefn.from_args(
    args.update(
        jjweb_reserve_info=JJWebPredictionInfo(
            reservation_expr=DEFAULT_RESERVATION_EXPR,
            force_po_prediction_for_reservation=True,
        ),
        use_old_current_period_method=False,
        extra_descriptor="_main",
    )
)


analysis_defn_3pl_east = FbaRevDefn.from_args(
    args.update(
        jjweb_reserve_info=JJWebPredictionInfo(
            reservation_expr=DEFAULT_RESERVATION_EXPR,
            force_po_prediction_for_reservation=True,
        ),
        use_old_current_period_method=False,
        extra_descriptor="3pl_east",
    )
)
