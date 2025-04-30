"""Information used  in order to set up and/or execute analyses."""

from __future__ import annotations

from jjpred.analysisdefn import (
    DEFAULT_RESERVATION_EXPR,
    CurrentSeasonDefn,
    FbaRevDefn,
    JJWebPredictionInfo,
)
from jjpred.inputstrategy import RefillType

from jjpred.analysisconfig import RefillConfigInfo

current_seasons = CurrentSeasonDefn(FW=24, SS=25)
analysis_date = "2025-APR-23"
dispatch_date = "2025-APR-21"
master_sku_date = "2025-APR-23"
sales_and_inventory_date = "2025-APR-22"
warehouse_inventory_date = "2025-APR-23"
website_sku_date = "2025-MAR-18"
config_date = "2025-APR-07"
in_stock_ratio_date = "2025-APR-01"
prediction_type_meta_date = None
check_dispatch_date = False
mainprogram_date = None  # "2025-FEB-25"
refill_draft_date = None  # "2025-FEB-25"

extra_refill_config_info: list[RefillConfigInfo] = [
    RefillConfigInfo(["amazon.ca", "amazon.com"], 5, "HCA0-SND-S")
]
combine_hca0_hcb0_gra_asg_history: bool = True

analysis_defn = FbaRevDefn(
    analysis_date=analysis_date,
    dispatch_date=dispatch_date,
    master_sku_date=master_sku_date,
    sales_and_inventory_date=sales_and_inventory_date,
    warehouse_inventory_date=warehouse_inventory_date,
    config_date=config_date,
    in_stock_ratio_date=in_stock_ratio_date,
    prediction_type_meta_date=prediction_type_meta_date,
    refill_type=RefillType.WEEKLY,
    mainprogram_date=mainprogram_date,  # "2025-FEB-18",
    refill_draft_date=refill_draft_date,  # "2025-FEB-18",
    mon_sale_r_date=None,
    po_date=None,
    new_overrides_e=True,
    match_main_program_month_fractions=True,
    check_dispatch_date=check_dispatch_date,
    extra_refill_config_info=extra_refill_config_info,
    combine_hca0_hcb0_gra_asg_history=combine_hca0_hcb0_gra_asg_history,
    current_seasons=current_seasons,
)

analysis_defn_website_reserved = FbaRevDefn(
    analysis_date=analysis_date,
    dispatch_date=dispatch_date,
    master_sku_date=master_sku_date,
    sales_and_inventory_date=sales_and_inventory_date,
    warehouse_inventory_date=warehouse_inventory_date,
    config_date=config_date,
    in_stock_ratio_date=in_stock_ratio_date,
    prediction_type_meta_date=prediction_type_meta_date,
    website_sku_date=website_sku_date,
    jjweb_reserve_info=JJWebPredictionInfo(
        reservation_expr=DEFAULT_RESERVATION_EXPR,
        force_po_prediction_for_reservation=True,
    ),
    refill_type=RefillType.WEEKLY,
    mainprogram_date=mainprogram_date,
    refill_draft_date=refill_draft_date,
    mon_sale_r_date=None,
    po_date=None,
    new_overrides_e=True,
    match_main_program_month_fractions=True,
    check_dispatch_date=check_dispatch_date,
    extra_descriptor="_website_reserved",
    extra_refill_config_info=extra_refill_config_info,
    combine_hca0_hcb0_gra_asg_history=combine_hca0_hcb0_gra_asg_history,
    current_seasons=current_seasons,
)
