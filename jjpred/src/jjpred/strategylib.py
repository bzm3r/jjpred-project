"""Where different input strategies are defined for use in analyses."""

from __future__ import annotations
from calendar import Month

import polars as pl

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import total_ordering

from analysis_tools.utils import get_analysis_defn_and_db
from jjpred.aggregator import (
    Aggregator,
    UsingAllChannels,
    UsingCanUSRetail,
    UsingRetail,
)
from jjpred.analysisdefn import AnalysisDefn, RefillDefn
from jjpred.channel import Channel
from jjpred.database import DataBase
from jjpred.inputstrategy import InputStrategy, SimpleTimePeriod
from jjpred.sku import Category

from jjpred.utils.datetime import Date
from jjpred.utils.multidict import MultiDict


@total_ordering
@dataclass
class StrategyId:
    name: str = field(compare=False)
    # date: Date = field(compare=False)
    hash: int = field(init=False, compare=False)

    def __init__(self, name: str):  # , date: DateLike):
        self.name = name
        self.hash = str(self).__hash__()
        # self.date = Date.from_datelike(date)

    def __str__(self) -> str:
        return f"{self.name}"  # _{str(self.date)}"

    def __hash__(self) -> int:
        return self.hash

    def __lt__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.name < other.name

        return False

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.name == other.name

        return False


LATEST = StrategyId("latest")
AUG_BIG = StrategyId("aug_big")
PO_DOUBLE_CHECK = StrategyId("po_double_check")

ALL_CHANNEL_AGGREGATOR = UsingAllChannels()
ALL_CAN_US_RETAIL_AGGREGATOR = UsingCanUSRetail()
AMAZON_CA_AGGREGATOR = UsingRetail(["Amazon.ca"])

# # generated using mainprogram_current_period_defn.ipynb
# CURRENT_PERIODS: MultiDict[Category, SimpleTimePeriod] = MultiDict(
#     data={
#         (
#             "AAA",
#             "ACA",
#             "ACB",
#             "AHJ",
#             "AJP",
#             "AJS",
#             "BSL",
#             "GBX",
#             "GHA",
#             "GUA",
#             "GUX",
#             "HAD0",
#             "HAV0",
#             "HBS",
#             "HBU",
#             "HCA0",
#             "HCB0",
#             "HCF0",
#             "HJP",
#             "HJS",
#             "HLC",
#             "HLH",
#             "HXC",
#             "HXP",
#             "HXU",
#             "SBS",
#             "SJD",
#             "SJF",
#             "SKB",
#             "SPW",
#             "SSS",
#             "UG1",
#             "UJ1",
#             "USA",
#             "UST",
#             "UT1",
#             "UV2",
#             "UVT",
#         ): UndeterminedSimpleTimePeriod("2025-MAR-01"),
#         ("BSL",): UndeterminedSimpleTimePeriod("2024-JUL-01"),
#         (
#             "WPS",
#             "WSS",
#             "WMT",
#             "WRM",
#             "WJA",
#             "WPF",
#             "BRC",
#             "BTB",
#             "BTL",
#             "BSW",
#             "BSA",
#             "IHT",
#             "KMT",
#             "SKT",
#             "WJT",
#             "WSF",
#             "WBF",
#             "WBS",
#             "WGS",
#             "BTT",
#             "BCV",
#             "FJM",
#             "FPM",
#             "FSM",
#             "FMR",
#             "LBS",
#             "LAN",
#             "KEH",
#             "AWWJ",
#             "LAB",
#             "SWS",
#             "ICP",
#             "IPC",
#             "IPS",
#             "ISS",
#             "ISB",
#             "AJA",
#             "SSW",
#             "LBP",
#             "LBT",
#             "BST",
#             "FAN",
#             "FHA",
#             "ISJ",
#         ): UndeterminedSimpleTimePeriod("2024-SEP-01"),
#         ("XBM", "XBK", "XLB", "XPC"): UndeterminedSimpleTimePeriod(
#             "2024-AUG-01"
#         ),
#         ("XWG",): UndeterminedSimpleTimePeriod("2024-NOV-01"),
#     }
# )
# """Current period definitions based on the main program file."""

# from historic_extract.ipynb
REFERENCE_CATEGORIES: MultiDict[Category, Category] = MultiDict(
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
)
"""Reference category definitions for SKUs, determined from historical
(containing monthly sales ratio sheet) Excel file."""


# from historic_extract.ipynb
PER_CHANNEL_REFERENCE_CHANNELS: Mapping[
    Channel, Mapping[Category, Aggregator] | Aggregator
] = {
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
            ("FMR", "KEH", "BSW", "BSA", "BRC", "KMT"): AMAZON_CA_AGGREGATOR,
        }
    ).as_dict(),
    Channel.parse("Wholesale"): MultiDict(
        data={
            ("XBM", "LBS", "FPM", "BCV"): ALL_CAN_US_RETAIL_AGGREGATOR,
            ("UST", "HBU", "HLC", "HXC", "HXU", "HXP"): AMAZON_CA_AGGREGATOR,
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
}
"""Sometimes, some channels use the same aggregation strategy (i.e. the sames
historical sales data) as another channel, rather than the default of using
their own channel's data."""

# # When setting the default value for the current period (see below), we are
# # making the assumption: 2024-MAR to *now*, and this has to be less than 12
# # months in length!
# assert datetime.datetime.today().month <= 3


def get_default_current_period_dict(
    analysis_defn_or_db: AnalysisDefn | DataBase,
) -> dict[str, SimpleTimePeriod]:
    analysis_defn, db = get_analysis_defn_and_db(analysis_defn_or_db)
    category_current_period_start_end = (
        db.meta_info.active_sku.select("category", "season")
        .unique()
        .with_columns(
            fw_start=pl.date(
                analysis_defn.current_seasons.FW + 2000,
                int(Month.SEPTEMBER),
                1,
            ),
            ss_start=pl.date(
                analysis_defn.current_seasons.SS + 2000, int(Month.MARCH), 1
            ),
        )
        .with_columns(
            start_date=pl.when(pl.col.season.eq("SS"))
            .then(pl.col.ss_start)
            .when(pl.col.season.eq("FW"))
            .then(pl.col.fw_start)
            .when(pl.col.season.eq("AS"))
            .then(pl.max_horizontal(pl.col.ss_start, pl.col.fw_start))
        )
        .with_columns(
            end_date=(
                analysis_defn.latest_dates.sales_history_latest_date
            ).as_polars_date(),
        )
        .with_columns(
            start_date=pl.when(pl.col.end_date.lt(pl.col.start_date))
            .then(pl.col.start_date.dt.offset_by("-1y"))
            .otherwise(pl.col.start_date)
        )
    )

    current_period_dict = {
        str(x["category"]): SimpleTimePeriod(x["start_date"], x["end_date"])
        for x in category_current_period_start_end.to_dicts()
    }

    return current_period_dict


def get_last_year_as_current_period_dict(
    analysis_defn_or_db: RefillDefn | DataBase,
) -> dict[str, SimpleTimePeriod]:
    analysis_defn, db = get_analysis_defn_and_db(analysis_defn_or_db)
    assert isinstance(analysis_defn, RefillDefn)

    assert analysis_defn.dispatch_date >= Date.from_ymd(
        2005, Month.SEPTEMBER, 1
    )

    category_current_period_start_end = (
        db.meta_info.active_sku.select("category", "season")
        .unique()
        .with_columns(
            start_date=analysis_defn.date.as_polars_date()
            .dt.month_start()
            .dt.offset_by("-1y"),
        )
        .with_columns(
            end_date=pl.col.start_date.dt.offset_by("1y"),
        )
    )

    current_period_dict = {
        str(x["category"]): SimpleTimePeriod(x["start_date"], x["end_date"])
        for x in category_current_period_start_end.to_dicts()
    }

    return current_period_dict


def get_strategy_from_library(
    analysis_defn_or_db: RefillDefn | DataBase,
    id: StrategyId,
    current_period_overrides: dict[str, SimpleTimePeriod] | None = None,
) -> list[InputStrategy]:
    analysis_defn, db = get_analysis_defn_and_db(analysis_defn_or_db)

    current_period_dict = (
        get_default_current_period_dict(db)
        if current_period_overrides is None
        else current_period_overrides
    )

    assert isinstance(analysis_defn, RefillDefn)

    STRATEGY_LIBRARY: dict[StrategyId, list[InputStrategy]] = {
        LATEST: [
            InputStrategy(
                Channel.from_str(channel),
                REFERENCE_CATEGORIES.as_dict(),
                current_period_dict,
                PER_CHANNEL_REFERENCE_CHANNELS,
            )
            for channel in [
                "amazon.com",
                "amazon.ca",
                "janandjul.com",
                "jjweb ca east",
                "wholesale",
                "amazon.co.uk",
                "amazon.de",
            ]
        ],
    }

    try:
        return STRATEGY_LIBRARY[id]
    except KeyError as e:
        raise KeyError(
            f"Strategy {id} not found. Available strategies are:"
            f" {sorted(STRATEGY_LIBRARY.keys())}"
        ) from e
