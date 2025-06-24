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

from jjpred.analysisconfig import GeneralRefillConfigInfo, RefillConfigInfo

import polars as pl

args = FbaRevDefnArgs(
    analysis_date="2025-JUN-09",
    current_seasons=CurrentSeasonDefn(FW=24, SS=25),
    dispatch_date="2025-JUN-09",
    master_sku_date="2025-JUN-09",
    sales_and_inventory_date="2025-JUN-09",
    warehouse_inventory_date="2025-JUN-09",
    in_stock_ratio_date="2025-JUN-02",
    website_sku_date="2025-MAR-18",
    config_date="2025-APR-07",
    prediction_type_meta_date=None,
    check_dispatch_date=False,
    mainprogram_date=None,  # "2025-FEB-25",
    refill_draft_date=None,  # "2025-FEB-25",
    extra_refill_config_info=(
        [
            RefillConfigInfo(["amazon.ca", "amazon.com"], 5, "HCA0-SND-S"),
            GeneralRefillConfigInfo(
                ["amazon.ca", "amazon.com"], 10, pl.col.category.eq("BSL")
            ),
            GeneralRefillConfigInfo(
                ["amazon.ca", "amazon.com"], 10, pl.col.category.eq("SWS")
            ),
            GeneralRefillConfigInfo(
                ["amazon.ca", "amazon.com"], 10, pl.col.category.eq("SMF")
            ),
        ]
    ),
    combine_hca0_hcb0_gra_asg_history=True,
    refill_type=RefillType.WEEKLY,
    match_main_program_month_fractions=False,
    mon_sale_r_date=None,
    po_date=None,
    new_overrides_e=True,
)

analysis_defn_no_reservation = FbaRevDefn.from_args(args)

analysis_defn_website_reserved_force_po_new_method = FbaRevDefn.from_args(
    args.update(
        jjweb_reserve_info=JJWebPredictionInfo(
            reservation_expr=DEFAULT_RESERVATION_EXPR,
            force_po_prediction_for_reservation=True,
        ),
        use_old_current_period_method=False,
        extra_descriptor="web_res_force_po_new_rolling_update_v2_new_type",
    )
)


analysis_defn_website_reserved_force_po_old_method = FbaRevDefn.from_args(
    args.update(
        jjweb_reserve_info=JJWebPredictionInfo(
            reservation_expr=DEFAULT_RESERVATION_EXPR,
            force_po_prediction_for_reservation=True,
        ),
        use_old_current_period_method=True,
        extra_descriptor="web_res_force_po_new_rolling_update_v2_old_type",
    )
)
