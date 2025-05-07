"""Information used in order to set up and/or execute trends analyses."""

from __future__ import annotations

from jjpred.analysisdefn import AnalysisDefn, CurrentSeasonDefn

current_seasons = CurrentSeasonDefn(FW=24, SS=25)

analysis_defn = AnalysisDefn(
    basic_descriptor="trend_analysis",
    date="2025-APR-01",
    master_sku_date="2025-APR-01",
    sales_and_inventory_date="2025-APR-01",
    warehouse_inventory_date="2025-APR-01",
    config_date="2025-FEB-25",
    in_stock_ratio_date="2025-APR-01",
    current_seasons=current_seasons,
)
