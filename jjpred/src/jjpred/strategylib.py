"""Where different input strategies are defined for use in analyses."""

from __future__ import annotations

from calendar import month_abbr
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import total_ordering

from jjpred.aggregator import (
    Aggregator,
    UsingAllChannels,
    UsingCanUSRetail,
    UsingRetail,
)
from jjpred.analysisdefn import RefillDefn
from jjpred.channel import Channel
from jjpred.inputstrategy import (
    InputStrategy,
    ContiguousTimePeriod,
    UndeterminedTimePeriod,
)
from jjpred.sku import Category

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

# generated using mainprogram_current_period_defn.ipynb
CURRENT_PERIODS: MultiDict[
    Category, ContiguousTimePeriod | UndeterminedTimePeriod
] = MultiDict(
    data={
        # (
        #     "HCF0",
        #     "HCB0",
        #     "HCA0",
        #     "HAD0",
        #     "HAV0",
        #     "HBS",
        #     "HXP",
        #     "HXC",
        #     "HXU",
        #     "HBU",
        #     "HLH",
        #     "GUX",
        #     "GUA",
        #     "GBX",
        #     "GHA",
        #     "SKB",
        #     "SKG",
        #     "SJF",
        #     "SJD",
        #     "SKX",
        #     "SMF",
        #     "HJS",
        #     "UJ1",
        #     "UT1",
        #     "UV2",
        #     "UG1",
        #     "USA",
        #     "AAA",
        #     "ACA",
        #     "ACB",
        #     "AJS",
        #     "AHJ",
        #     "SPW",
        # ): TimePeriod("2025-JAN-01", "2025-APR-01"),
        ("BSL",): UndeterminedTimePeriod("2024-JUL-01"),
        (
            "WPS",
            "WSS",
            "WMT",
            "WRM",
            "WJA",
            "WPF",
            "BRC",
            "BTB",
            "BTL",
            "BSW",
            "BSA",
            "IHT",
            "KMT",
            "SKT",
            "WJT",
            "WSF",
            "WBF",
            "WBS",
            "WGS",
            "BTT",
            "BCV",
            "FJM",
            "FPM",
            "FSM",
            "FMR",
            "LBS",
            "LAN",
            "KEH",
            "AWWJ",
            "LAB",
            "SWS",
            "ICP",
            "IPC",
            "IPS",
            "ISS",
            "ISB",
            "AJA",
            "SSW",
            "LBP",
            "LBT",
            "BST",
            "FAN",
            "FHA",
            "ISJ",
        ): UndeterminedTimePeriod("2024-SEP-01"),
        ("XBM", "XBK", "XLB", "XPC"): UndeterminedTimePeriod("2024-AUG-01"),
        ("XWG",): UndeterminedTimePeriod("2024-NOV-01"),
    }
)
"""Current period definitions based on the main program file."""

# from historic_extract.ipynb
REFERENCE_CATEGORIES: MultiDict[Category, Category] = MultiDict(
    data={
        ("AJA",): "WJT",
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
        ("AWWJ", "WJT"): "WJA",
        ("WGS", "WRM"): "WMT",
        ("UST",): "UT1",
        ("HBU",): "HBS",
        ("HLC", "HXC", "HXU"): "HXP",
        ("HJS", "AJS", "ACB", "ACA", "AAA", "HLH"): "HCF0",
        ("SPW", "SJF"): "SKB",
    }
)
"""Reference category definitions for SKUs, determined from historical
(containing monthly sales ratio sheet) Excel file."""


# from historic_extract.ipynb
PER_CHANNEL_REFERENCE_CHANNELS: Mapping[
    Channel, Mapping[Category, Aggregator] | Aggregator
] = {
    Channel.parse("Amazon US RETAIL"): MultiDict(
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
            ): ALL_CHANNEL_AGGREGATOR,
            ("FMR", "KEH", "BSW", "BSA", "BRC", "KMT"): AMAZON_CA_AGGREGATOR,
        }
    ).as_dict(),
    Channel.parse("Wholesale ALL_REGION WHOLESALE"): MultiDict(
        data={
            ("XBM", "LBS", "FPM", "BCV"): ALL_CHANNEL_AGGREGATOR,
            ("UST", "HBU", "HLC", "HXC", "HXU", "HXP"): AMAZON_CA_AGGREGATOR,
        }
    ).as_dict(),
    Channel.parse("Amazon UK RETAIL"): MultiDict(
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
    Channel.parse("JanAndJul CA|US RETAIL"): ALL_CAN_US_RETAIL_AGGREGATOR,
    # Channel.parse("JanAndJul CA|US RETAIL"): MultiDict(
    #     data={
    #         (
    #             "XBK",
    #             "XBM",
    #             "LBS",
    #             "FPM",
    #             "BCV",
    #             "UT1",
    #             "USA",
    #             "UG1",
    #             "UJ1",
    #             "UV2",
    #         ): ALL_CAN_US_RETAIL_AGGREGATOR,
    #         (
    #             "FMR",
    #             "KEH",
    #             "WSF",
    #             "BSW",
    #             "BSA",
    #             "BRC",
    #             "BTB",
    #             "KMT",
    #             "WMT",
    #             "WSS",
    #             "UST",
    #             "HBU",
    #             "HLC",
    #             "HXC",
    #             "HXU",
    #             "HXP",
    #             "GUX",
    #             "HBS",
    #             "SKG",
    #             "SPW",
    #             "SJF",
    #         ): AMAZON_CA_AGGREGATOR,
    #     }
    # ).as_dict(),
    Channel.parse("Amazon DE RETAIL"): MultiDict(
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
    Channel.parse("Amazon CA RETAIL"): MultiDict(
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
            ): ALL_CHANNEL_AGGREGATOR
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


def get_strategy_from_library(
    analysis_defn: RefillDefn, id: StrategyId
) -> list[InputStrategy]:
    default_time_period = UndeterminedTimePeriod(
        f"{analysis_defn.dispatch_date.year - 1}-{month_abbr[analysis_defn.dispatch_date.month].upper()}"
    )
    print(f"{default_time_period}: {default_time_period.start}")

    STRATEGY_LIBRARY: dict[StrategyId, list[InputStrategy]] = {
        LATEST: [
            InputStrategy(
                Channel.from_str(channel),
                REFERENCE_CATEGORIES.as_dict(),
                defaultdict(
                    lambda: default_time_period,
                    CURRENT_PERIODS.as_dict(),
                ),
                PER_CHANNEL_REFERENCE_CHANNELS,
            )
            for channel in [
                "amazon.com",
                "amazon.ca",
                "janandjul.com",
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
