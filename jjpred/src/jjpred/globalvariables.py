"""Global variables used to define the main parameters of an analysis."""
# TODO: this is a temporary hack. Should improve it by making the entire
# analysis a function.

from __future__ import annotations

from jjpred.utils.datetime import DateOffset, DateUnit

DEFAULT_STORAGE_FORMAT: str = "parquet"
"""Use ``parquet`` files for default storage as they are robust (unlike the
``arrow`` format) for saving Polars data to  and compact."""

WEEKLY_PREDICTION_OFFSET: DateOffset = DateOffset(6, DateUnit.WEEK)
"""For a weekly refill, we offset 6 weeks into the future."""

SEASON_START_PREDICTION_OFFSET: DateOffset = DateOffset(3, DateUnit.MONTH)
"""For a season start refill, we offset 3 months into the future."""

MAIN_VS_THIS_TOLERANCE: int = 0
"""If the difference in dispatch for the main program vs. this program is equal
to or less than this many units, then we consider it to be okay."""

DISPATCH_CUTOFF_QTY: int = 2
"""Minimum number of items to dispatch for an FBA refill."""

LOW_CURRENT_PERIOD_SALES: int = 20
"""If current period sales of SKU are lower than this value, then mark the SKU
as having low current period sales (relevant if historical sales data + current
period data is used to generate a prediction for expected sales for this
SKU)."""

LOW_CATEGORY_HISTORICAL_SALES: int = 100
"""If category historical sales of SKU are lower than this value, then mark the SKU
as having low category historical sales (relevant if historical sales data + current
period data is used to generate a prediction for expected sales for this
SKU)."""

# This is read by main_program_new_categories.ipynb
# also manually added: FVM, AJC, SBS, but this should correspond to changes in
# the main program
NEW_CATEGORIES = [
    "AJA",
    "AJC",
    "BSL",
    "BST",
    "DRC",
    "FAN",
    "FHA",
    "FVM",
    "IHT",  # temporarily adding IHT
    "IPC",
    "IPS",
    "ISB",
    "ISJ",
    "ISS",
    "LAB",
    "LAN",
    "LBP",
    "LBT",
    # "SKX",
    "SBS",
    "SMF",
    "SSW",
    "SWS",
    "WGS",
    "WRM",
    "XBK",
    "XBM",
    "XLB",
    "XPC",
    "XWG",
]
"""Categories marked "NEW" as per marketing, so we use "NE" type prediction for
these, if they are in the "PO"-based prediction phase of the year."""

ENDS_WITH_U_OK_LIST = ["HXU", "HBU"]
"""We typically ignore SKUs that end with U, but some should not be ignored."""

STARTS_WITH_M_OK_LIST = []
"""We typically ignore SKUs that start with M, but some should not be
ignored."""

IGNORE_CATEGORY_LIST = [
    # "DEAL-MISC-KCS",
    # "DEAL-MISC",
    # "STORE",
    # "GUB-AST",
    # "GUB-BOX",
    # "SKB-INSOL",
    # "COHY",
]
"""We ignore SKUs with ``category`` in the above list for the purposes of
FBA refill."""

IGNORE_SKU_LIST = [
    "HCB0-WHA-XL",
    "HCB0-WHA-S",
    "HCB0-WHA-L",
    "HCB0-WHA-M",
    "SGL-WHA-M",
]
"""These SKUs are in history files, but are not in the Master SKU file."""

OUTPERFORM_FACTOR: float = 0.2
"""If an item's historical estimate is greater than ``1 + OUTPERFORM_FACTOR``
of the PO estimate, then we consider it to be an "outperformer". If it is less
than ``1 - OUTPERFORM_FACTOR`` then we consider it to be an "underperformer". If
an item does not have a PO estimate to bench mark against, we cannot do such
benchmarking."""
