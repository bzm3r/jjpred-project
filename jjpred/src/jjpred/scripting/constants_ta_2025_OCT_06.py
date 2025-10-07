"""Information used in order to set up and/or execute trends analyses."""

from __future__ import annotations

from jjpred.analysisdefn import AnalysisDefn, CurrentSeasonDefn

current_seasons = CurrentSeasonDefn(FW=25, SS=25)

analysis_defn = AnalysisDefn(
    basic_descriptor="trend_analysis",
    date="2025-OCT-06",
    master_sku_date="2025-OCT-06",
    sales_and_inventory_date="2025-OCT-06",
    warehouse_inventory_date="2025-OCT-06",
    config_date="2025-SEP-29",
    in_stock_ratio_date="2025-OCT-06",
    current_seasons=current_seasons,
)
