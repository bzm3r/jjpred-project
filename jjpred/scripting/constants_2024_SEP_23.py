"""Information used  in order to set
up and/or execute analyses."""

from __future__ import annotations

from jjpred.analysisdefn import FbaRevDefn
from jjpred.inputstrategy import RefillType
from jjpred.utils.datetime import Date, DateLike


# analysis_date: DateLike = "2024-SEP-23"
# dispatch_date: DateLike = Date.from_datelike("2024-SEP-01")
# real_analysis_date: DateLike = "2024-SEP-09"
# config_date: DateLike = "2024-SEP-09"
# master_sku_date: DateLike = real_analysis_date
# historical_sales_date: DateLike = real_analysis_date
# inventory_date: DateLike = real_analysis_date
# mainprogram_date: DateLike = real_analysis_date
# calc_file_date: DateLike = real_analysis_date
# refill_draft_date: DateLike = real_analysis_date
# analysis_defn = FbaRevDefn(dispatch_date, analysis_date)
# refill_type = RefillType.CUSTOM_2024_SEP_10
# prediction_type_meta: str | DateLike | None = Date.from_datelike("2024-SEP-23")

analysis_defn = FbaRevDefn.new_comparison_analysis(
    "2024-SEP-23",
    "2024-SEP-02",
    "2024-SEP-09",
    "2024-SEP-23",
    "2024-SEP-09",
    RefillType.CUSTOM_2024_SEP_10,
)
