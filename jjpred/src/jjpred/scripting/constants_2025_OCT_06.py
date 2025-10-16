"""Information used  in order to set up and/or execute analyses."""

from __future__ import annotations
from calendar import Month

from jjpred.aggregator import UsingAllChannels, UsingCanUSRetail, UsingRetail
from jjpred.analysisdefn import (
    CurrentSeasonDefn,
    RefillDefn,
    RefillDefnArgs,
    JJWebPredictionInfo,
)
from jjpred.channel import Channel
from jjpred.inputstrategy import RefillType

from jjpred.analysisconfig import GeneralRefillConfigInfo

import polars as pl

from jjpred.utils.multidict import MultiDict

ALL_CHANNEL_AGGREGATOR = UsingAllChannels()
ALL_CAN_US_RETAIL_AGGREGATOR = UsingCanUSRetail()
AMAZON_CA_AGGREGATOR = UsingRetail(["Amazon.ca"])


FW_RESERVATION_MONTHS = [(Month.JULY.value, Month.JANUARY.value)]
"""If the item is FW: then the reservation period is: Aug to Jan (not inclusive)."""
SS_TYPICAL_RESERVATION_MONTHS = [(Month.FEBRUARY.value, Month.JUNE.value)]
"""If the item is SS: then typically the reservation period is: Feb to Jun (not
inclusive)."""
SS_SPW_AND_U_CATS_RESERVATION_MONTHS = [
    (Month.FEBRUARY.value, Month.JULY.value)
]
"""If the item is SS, and its category is SPW or one of the U* categories, then
typically the reservation period is: Feb to Jul (not inclusive)."""

DEFAULT_RESERVATION_EXPR = (
    pl.when(pl.col.season.eq("FW"))
    .then(FW_RESERVATION_MONTHS)
    .when(
        pl.col.season.eq("SS")
        & ~(
            pl.col.category.eq("SPW")
            | pl.col.category.cast(pl.String()).str.starts_with("U")
        )
    )
    .then(SS_TYPICAL_RESERVATION_MONTHS)
    .when(
        pl.col.season.eq("SS")
        & (
            pl.col.category.eq("SPW")
            | pl.col.category.cast(pl.String()).str.starts_with("U")
        )
    )
    .then(SS_SPW_AND_U_CATS_RESERVATION_MONTHS)
    .when(pl.col.season.eq("AS"))
    .then(SS_TYPICAL_RESERVATION_MONTHS + FW_RESERVATION_MONTHS)
    # months should be a pair of UInt8
    .cast(pl.List(pl.Array(pl.UInt8(), 2)))
)
"""The default reservation expression. Built using ``FW_RESERVATION_MONTHS``,
``SS_TYPICAL_RESERVATION_MONTHS`` and ``SS_SPW_AND_U_CAT_RESERVATION_MONTHS``.
"""

args = RefillDefnArgs(
    refill_description="refill",
    analysis_date="2025-OCT-06",
    current_seasons=CurrentSeasonDefn(FW=25, SS=25),
    dispatch_date="2025-OCT-06",
    master_sku_date="2025-OCT-06",
    sales_and_inventory_date="2025-OCT-06",
    warehouse_inventory_date="2025-OCT-06",
    in_stock_ratio_date="2025-OCT-06",
    website_sku_date="2025-SEP-18",
    config_date="2025-SEP-29",
    ignore_sku_list=[
        "HCB0-WHA-XL",
        "HCB0-WHA-S",
        "HCB0-WHA-L",
        "HCB0-WHA-M",
        "SGL-WHA-M",
    ],
    ignore_category_list=[],
    prediction_type_meta_date=None,
    check_dispatch_date=False,
    mainprogram_date=None,  # "2025-FEB-25",
    refill_draft_date=None,  # "2025-FEB-25",
    match_main_program_month_fractions=False,
    extra_refill_config_info=(
        [
            GeneralRefillConfigInfo(
                ["amazon.ca", "amazon.com"], 10, pl.col.category.eq("BSL")
            ),
            GeneralRefillConfigInfo(
                ["amazon.ca", "amazon.com"],
                10,
                pl.col.category.eq("SMF") & pl.col.print.is_in(["DPK", "SBR"]),
            ),
            GeneralRefillConfigInfo(
                ["amazon.ca", "amazon.com"],
                5,
                pl.col.min_refill_request.lt(5) & pl.col.season.ne("SS"),
            ),
        ]
    ),
    combine_hca0_hcb0_gra_asg_history=True,
    refill_type=RefillType.END_OF_DEC_2025,
    mon_sale_r_date=None,
    po_date=None,
    new_overrides_e=True,
    forced_po_categories=["IHT"],
    additional_new_categories=["FVM", "ISJ", "SBS", "SMF"],
    reference_categories=MultiDict(
        data={
            ("AJA", "AWWJ"): "WJT",
            ("WPO", "WJO", "FSM", "FJM"): "FPM",
            ("GBX",): "GUX",
            ("GHA",): "GUA",
            ("ISJ",): "ISS",
            ("FHA", "LAB"): "KEH",
            ("FAN",): "LAN",
            ("BST", "BTT"): "BTB",
            ("IPS", "IPC", "ISS", "ISB", "ICP"): "IHT",
            ("XWG", "WBS"): "WPS",
            ("XLB",): "XBM",
            ("XPC",): "XBK",
            ("LBP", "LBT"): "LAB",
            ("SMF", "SWS"): "SKG",
            ("LAN", "WBF"): "WPF",
            ("WGS", "WRM"): "WMT",
            ("UST",): "UT1",
            ("HBU",): "HBS",
            ("HLC", "HXC", "HXU"): "HXP",
            ("HJS", "AJS", "ACB", "ACA", "AAA", "HLH"): "HCF0",
            ("BSL",): "BSA",
        }
    ).as_dict(),
    per_channel_reference_channels={
        Channel.parse("Amazon US"): MultiDict(
            data={
                (
                    "XBK",
                    "XBM",
                    "LBS",
                    "FPM",
                    "BCV",
                    "UT1",
                    "USA",
                    "UG1",
                    "UJ1",
                    "UV2",
                    "HXP",
                ): ALL_CAN_US_RETAIL_AGGREGATOR,
                (
                    "FMR",
                    "KEH",
                    "BSW",
                    "BSA",
                    "BRC",
                    "KMT",
                ): AMAZON_CA_AGGREGATOR,
            }
        ).as_dict(),
        Channel.parse("Wholesale"): MultiDict(
            data={
                ("XBM", "LBS", "FPM", "BCV"): ALL_CAN_US_RETAIL_AGGREGATOR,
                (
                    "UST",
                    "HBU",
                    "HLC",
                    "HXC",
                    "HXU",
                    "HXP",
                ): AMAZON_CA_AGGREGATOR,
            }
        ).as_dict(),
        Channel.parse("Amazon UK"): MultiDict(
            data={
                (
                    "AJA",
                    "WPO",
                    "WJO",
                    "GBX",
                    "GHA",
                    "ISJ",
                    "FHA",
                    "FAN",
                    "BST",
                    "IPS",
                    "XWG",
                    "XLB",
                    "XPC",
                    "XBK",
                    "XBM",
                    "LBP",
                    "LBT",
                    "IPC",
                    "ISS",
                    "ISB",
                    "SMF",
                    "SWS",
                    "ICP",
                    "LBS",
                    "LAN",
                    "LAB",
                    "FSM",
                    "FPM",
                    "FJM",
                    "FMR",
                    "BTT",
                    "KEH",
                    "AWWJ",
                    "BCV",
                    "WGS",
                    "WBS",
                    "WBF",
                    "WSF",
                    "WJT",
                    "BSW",
                    "BSA",
                    "BRC",
                    "SKT",
                    "BTL",
                    "BTB",
                    "KMT",
                    "IHT",
                    "WRM",
                    "WMT",
                    "WSS",
                    "WPS",
                    "WPF",
                    "WJA",
                    "UST",
                    "HBU",
                    "HLC",
                    "HXC",
                    "HXU",
                    "HJS",
                    "AJS",
                    "ACB",
                    "ACA",
                    "AAA",
                    "UT1",
                    "USA",
                    "UG1",
                    "UJ1",
                    "UV2",
                    "HLH",
                    "HXP",
                    "GUA",
                    "GUX",
                    "HBS",
                    "SKX",
                    "SKG",
                    "SPW",
                    "SJF",
                    "SKB",
                ): AMAZON_CA_AGGREGATOR
            }
        ).as_dict(),
        Channel.parse("janandjul.com"): ALL_CAN_US_RETAIL_AGGREGATOR,
        Channel.parse("jjweb ca east"): UsingRetail(["jjweb ca east"]),
        Channel.parse("Amazon DE"): MultiDict(
            data={
                (
                    "AJA",
                    "WPO",
                    "WJO",
                    "GBX",
                    "GHA",
                    "ISJ",
                    "FHA",
                    "FAN",
                    "BST",
                    "IPS",
                    "XWG",
                    "XLB",
                    "XPC",
                    "XBK",
                    "XBM",
                    "LBP",
                    "LBT",
                    "IPC",
                    "ISS",
                    "ISB",
                    "SMF",
                    "SWS",
                    "ICP",
                    "LBS",
                    "LAN",
                    "LAB",
                    "FSM",
                    "FPM",
                    "FJM",
                    "FMR",
                    "BTT",
                    "KEH",
                    "AWWJ",
                    "BCV",
                    "WGS",
                    "WBS",
                    "WBF",
                    "WSF",
                    "WJT",
                    "BSW",
                    "BSA",
                    "BRC",
                    "SKT",
                    "BTL",
                    "BTB",
                    "KMT",
                    "IHT",
                    "WRM",
                    "WMT",
                    "WSS",
                    "WPS",
                    "WPF",
                    "WJA",
                    "UST",
                    "HBU",
                    "HLC",
                    "HXC",
                    "HXU",
                    "HJS",
                    "AJS",
                    "ACB",
                    "ACA",
                    "AAA",
                    "UT1",
                    "USA",
                    "UG1",
                    "UJ1",
                    "UV2",
                    "HLH",
                    "HXP",
                    "GUA",
                    "GUX",
                    "HBS",
                    "SKX",
                    "SKG",
                    "SPW",
                    "SJF",
                    "SKB",
                ): AMAZON_CA_AGGREGATOR,
            }
        ).as_dict(),
        Channel.parse("Amazon CA"): MultiDict(
            data={
                (
                    "XBK",
                    "XBM",
                    "LBS",
                    "FPM",
                    "BCV",
                    "UT1",
                    "USA",
                    "UG1",
                    "UJ1",
                    "UV2",
                    "HXP",
                ): ALL_CAN_US_RETAIL_AGGREGATOR
            }
        ).as_dict(),
    },
    enable_full_box_logic=True,
    full_box_rounding_margin_qty=10,
    full_box_rounding_margin_ratio=0.2,
    ss_start_month=Month.MARCH,
    fw_start_month=Month.AUGUST,
)

analysis_defn_fba = RefillDefn.from_args(
    args.update(
        jjweb_reserve_info=JJWebPredictionInfo(
            reservation_expr=DEFAULT_RESERVATION_EXPR,
            force_po_prediction_for_reservation=True,
        ),
        extra_descriptor="fba",
        channels=["amazon.ca", "amazon.com"],
    ),
)

analysis_defn_fba_test = RefillDefn.from_args(
    args.update(
        jjweb_reserve_info=JJWebPredictionInfo(
            reservation_expr=DEFAULT_RESERVATION_EXPR,
            force_po_prediction_for_reservation=True,
        ),
        extra_descriptor="fba_test",
        channels=["amazon.ca", "amazon.com"],
    ),
)

analysis_defn_3pl_east = RefillDefn.from_args(
    args.update(
        jjweb_reserve_info=JJWebPredictionInfo(
            reservation_expr=DEFAULT_RESERVATION_EXPR,
            force_po_prediction_for_reservation=True,
        ),
        extra_descriptor="3pl_east",
        channels=["jjweb ca east"],
    )
)
