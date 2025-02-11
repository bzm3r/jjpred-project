"""Information used  in order to set
up and/or execute analyses."""

from __future__ import annotations
import copy

from jjpred.analysisdefn import FbaRevDefn
from jjpred.inputstrategy import RefillType

analysis_defn = FbaRevDefn(
    analysis_date="2025-FEB-11",
    dispatch_date="2025-FEB-11",
    master_sku_date="2025-FEB-10",
    sales_and_inventory_date="2025-FEB-10",
    warehouse_inventory_date="2025-FEB-09",
    config_date="2025-FEB-03",
    in_stock_ratio_date="2025-FEB-04",
    refill_type=RefillType.WEEKLY,
    mainprogram_date=None,
    refill_draft_date=None,
    prediction_type_meta_date=None,
    mon_sale_r_date=None,
    po_date=None,
    new_overrides_e=True,
    check_dispatch_date=False,
    match_main_program_month_fractions=True,
)

analysis_defn_main = copy.deepcopy(analysis_defn)
analysis_defn_main.extra_descriptor = "main"

analysis_defn_new = copy.deepcopy(analysis_defn_main)
analysis_defn_new.extra_descriptor = "new"

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
