"""Information used  in order to set up and/or execute analyses."""

from __future__ import annotations

from jjpred.analysisdefn import FbaRevDefn, JJWebDefn, ReservationInfo
from jjpred.inputstrategy import RefillType

import polars as pl

from jjpred.analysisconfig import RefillConfigInfo

analysis_date = "2025-APR-14"
dispatch_date = "2025-APR-14"
master_sku_date = "2025-APR-14"
sales_and_inventory_date = "2025-APR-14"
warehouse_inventory_date = "2025-APR-14"
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
    jjweb_reserve_to_date=[
        ReservationInfo(
            (
                pl.col.category.eq("SPW")
                | pl.col.category.cast(pl.String()).str.starts_with("U")
            )
            & pl.col.season.is_in(["AS", "SS"]),
            "2025-JUL-01",
        ),
        ReservationInfo(
            ~(
                pl.col.category.eq("SPW")
                | pl.col.category.cast(pl.String()).str.starts_with("U")
            )
            & pl.col.season.is_in(["AS", "SS"]),
            "2025-JUN-01",
        ),
    ],
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
)

# analysis_defn_main = copy.deepcopy(analysis_defn)
# analysis_defn_main.extra_descriptor = "main"

# analysis_defn_new = copy.deepcopy(analysis_defn)
# analysis_defn_new.extra_descriptor = "new"

# analysis_defn_test = copy.deepcopy(analysis_defn)
# analysis_defn_test.extra_descriptor = "test"

jjweb_analysis_defn = JJWebDefn(
    analysis_date=analysis_date,
    dispatch_date=dispatch_date,
    end_date="2025-MAY-01",
    website_sku_date=website_sku_date,
    master_sku_date=master_sku_date,
    sales_and_inventory_date=sales_and_inventory_date,
    warehouse_inventory_date=warehouse_inventory_date,
    config_date=config_date,
    in_stock_ratio_date=in_stock_ratio_date,
    prediction_type_meta_date=prediction_type_meta_date,
    proportion_split_date="2025-FEB-13",
    check_dispatch_date=check_dispatch_date,
    extra_refill_config_info=extra_refill_config_info,
    combine_hca0_hcb0_gra_asg_history=combine_hca0_hcb0_gra_asg_history,
)


# analysis_defn = FbaRevDefn.new_comparison_analysis(
#     analysis_date="2025-FEB-03",
#     dispatch_date="2025-FEB-01",
#     config_date="2025-FEB-03",
#     prediction_type_meta_date=None,
#     real_analysis_date="2025-FEB-03",
#     refill_type=RefillType.CUSTOM_2025_FEB_03,
#     new_overrides_e=True,
#     check_dispatch_date=False,
#     match_main_program_month_fractions=True,
#     in_stock_ratio_date="2025-JAN-19",
#     extra_descriptor="po_reorg",
# )

# analysis_date: DateLike = "2024-OCT-15"
# dispatch_date: DateLike = Date.from_datelike("2024-OCT-15")
# real_analysis_date: DateLike = "2024-OCT-15"
# config_date: DateLike = "2024-SEP-09"
# master_sku_date: DateLike = real_analysis_date
# historical_sales_date: DateLike = real_analysis_date
# inventory_date: DateLike = real_analysis_date
# mainprogram_date: DateLike = real_analysis_date
# calc_file_date: DateLike = real_analysis_date
# refill_draft_date: DateLike = real_analysis_date
# analysis_defn = FbaRevDefn(dispatch_date, analysis_date)
# refill_type = RefillType.WEEKLY
# prediction_type_meta: str | DateLike | None = Date.from_datelike("2024-SEP-23")
# start_date_required_month_parts: int | None = 3
# end_date_required_month_parts: int | None = 3
