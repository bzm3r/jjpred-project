"""Our dataframes often contain common groups of columns, which are listed out
here for general use."""

from __future__ import annotations

from jjpred.channel import Channel
from jjpred.sku import Sku
from jjpred.structlike import MemberType


ALL_SKU_IDS = ["a_sku"] + Sku.members(MemberType.META)
WHOLE_SKU_IDS = ["sku"] + ["a_sku"]
CHANNEL_IDS = Channel.members()
SEASON_IDS = [
    "season",
    "sku_year_history",
    "category_year_history",
    "sku_latest_year",
    "sku_latest_po_season",
    "sku_current_year",
    "sku_current_po_season",
]
ALL_SKU_AND_CHANNEL_IDS = ALL_SKU_IDS + CHANNEL_IDS
ALL_IDS = ALL_SKU_IDS + CHANNEL_IDS + SEASON_IDS
PAUSE_PLAN_IDS = [
    "website_sku",
    "pause_plan",
]
STATUS_IDS = ["status", "orphan_sku"]
MASTER_PAUSE_FLAGS = [
    "is_active",
    "is_master_paused",
]
PAUSE_FLAGS = MASTER_PAUSE_FLAGS + [
    "is_config_paused",
]
DATA_AVAILABILITY_FLAGS = [
    "has_historical_data",
    "has_current_data",
    "low_current_period_sales",
    "has_e_data",
    "has_po_data",
]
NOVELTY_FLAGS = [
    "is_new_category",
    "is_current_category",
    "is_future_category",
    "is_current_sku",
    "is_future_sku",
    "is_new_sku",
]
GENERAL_CHECK_FLAG_IDS = [
    "missing_current_period_defn",
]
INV_AVAILABILITY_FLAGS = [
    "missing_wh_stock_info",
    "missing_ch_stock_info",
    "zero_wh_dispatchable",
]
DATA_PROBLEM_FLAGS = [
    "new_category_problem",
    "po_problem",
    "e_problem",
    "ce_problem",
    "new_print_problem",
]
PREDICTION_CHECK_FLAGS = [
    "refill_request_override",
    "uses_refill_request",
    "uses_ce",
    "uses_e",
    "uses_po",
    "uses_ne",
    "has_low_isr",
    "e_overrides_po",
    "po_overrides_e",
    "ce_forced_to_use_e",
    "ce_forced_to_use_po",
    "ce_uses_po",
    "ce_uses_e",
    "uses_overperformer_estimate",
    "demand_based_on_e",
    "demand_based_on_po",
    "no_expected_demand_info",
    "low_current_period_sales",
    "low_category_historical_sales",
]
DISPATCH_CHECK_FLAGS = [
    "no_qty_box_info",
    "required_gt_supply",
    "auto_split",
    "fine_auto_split",
    "num_closest_box",
    "dispatch_below_cutoff",
]
FINAL_CHECK_FLAGS = (
    PAUSE_FLAGS
    + DATA_PROBLEM_FLAGS
    + INV_AVAILABILITY_FLAGS
    + PREDICTION_CHECK_FLAGS
    + DISPATCH_CHECK_FLAGS
)
PREDICTION_MODE_INFO = [
    "prediction_type",
    "new_overrides_e",
    "category_marked_new",
    "category_type",
    "refers_to",
    "referred_by",
    "performance_flag",
]
HISTORICAL_PERIOD_INFO = ["category_historical_year_sales"]
CURRENT_PERIOD_INFO = [
    "current_period",
    "current_period_sales",
]
DEMAND_INFO = (
    HISTORICAL_PERIOD_INFO
    + CURRENT_PERIOD_INFO
    + [
        "mean_current_period_isr",
        "expected_demand_from_history",
        "expected_demand_from_po",
        "expected_demand_before_missing_po_consideration",
        "expected_demand_last_year",
        "applies_missing_po_consideration",
        "expected_demand",
        "refill_request",
    ]
)
INVENTORY_DATA_AVAILABILITY_FLAGS = ["no_wh_stock_info", "no_ch_stock_info"]
INVENTORY_INFO = [
    "wh_dispatchable",
    "ch_stock",
    "jjweb_inv_3pl",
    "jjweb_east_frac",
    "has_reservation",
    "reserved_before_on_order",
    "reserved",
    "wh_dispatchable_accounting_jjweb_west",
    "wh_dispatchable_accounting_jjweb_east",
    "reserved_west",
    "reserved_including_3pl",
]
DISPATCH_INFO = INVENTORY_INFO + [
    "requesting",
    "qty_box",
    "post_box_required",
    "pre_box_required",
    "dispatch",
]
DISPATCHABLE_PAUSED_DATA = (
    SEASON_IDS
    + PAUSE_PLAN_IDS
    + PAUSE_FLAGS
    + ["wh_dispatchable", "expected_demand", "refill_request", "dispatch"]
)
MAIN_PROGRAM_INFO = [
    "in_main_program",
]
