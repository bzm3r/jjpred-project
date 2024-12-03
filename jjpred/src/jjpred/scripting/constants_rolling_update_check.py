"""Information used  in order to set
up and/or execute analyses."""

from __future__ import annotations

from jjpred.analysisdefn import FbaRevDefn
from jjpred.inputstrategy import RefillType

analysis_defn = FbaRevDefn.new_comparison_analysis(
    "2024-OCT-21",
    "2024-OCT-21",
    "2024-OCT-16",
    "2024-SEP-23",
    "2024-OCT-21",
    RefillType.WEEKLY,
    extra_descriptor="RUCHECK",
)

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
