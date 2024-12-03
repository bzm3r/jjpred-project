"""Information used  in order to set
up and/or execute analyses."""

from __future__ import annotations

from jjpred.analysisdefn import FbaRevDefn
from jjpred.inputstrategy import RefillType
# from jjpred.utils.datetime import Date, DateLike


# analysis_date: DateLike = "2024-OCT-10"
# dispatch_date: DateLike = Date.from_datelike("2024-OCT-07")
# real_analysis_date: DateLike = "2024-OCT-07"
# config_date: DateLike = "2024-SEP-09"
# master_sku_date: DateLike = real_analysis_date
# historical_sales_date: DateLike = real_analysis_date
# inventory_date: DateLike = real_analysis_date
# mainprogram_date: DateLike = real_analysis_date
# calc_file_date: DateLike = real_analysis_date
# refill_draft_date: DateLike = real_analysis_date
# prediction_type_meta: str | DateLike | None = Date.from_datelike("2024-SEP-23")
# refill_type = RefillType.WEEKLY

analysis_defn = FbaRevDefn.new_comparison_analysis(
    "2024-OCT-10",
    "2024-OCT-07",
    "2024-SEP-09",
    "2024-SEP-23",
    "2024-OCT-07",
    RefillType.WEEKLY,
)
