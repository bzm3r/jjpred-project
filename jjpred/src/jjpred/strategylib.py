"""Where different input strategies are defined for use in analyses."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Self
import polars as pl

from jjpred.aggregator import (
    Aggregator,
    UsingAllChannels,
    UsingCanUSRetail,
    UsingRetail,
)
from jjpred.channel import Channel
from jjpred.globalvariables import OFF_SEASONS
from jjpred.inputstrategy import (
    InputStrategy,
    TimePeriod,
    RefillType,
    UndeterminedTimePeriod,
)
from jjpred.seasons import Season
from jjpred.sku import Category

from jjpred.utils.datetime import (
    Date,
    DateLike,
    Month,
)
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

ALL_CHANNEL_AGGREGATOR = UsingAllChannels()
ALL_CAN_US_RETAIL_AGGREGATOR = UsingCanUSRetail()
AMAZON_CA_AGGREGATOR = UsingRetail(["Amazon.ca"])

# generated using mainprogram_current_period_defn.ipynb
CURRENT_PERIODS: MultiDict[Category, TimePeriod | UndeterminedTimePeriod] = (
    MultiDict(
        data={
            (
                "HCF0",
                "HCB0",
                "HCA0",
                "HAD0",
                "HAV0",
                "HBS",
                "HXP",
                "HXC",
                "HXU",
                "GUX",
                "GUA",
                "GBX",
                "GHA",
                "SKB",
                "SKG",
                "SJF",
                "SJD",
                "SKX",
                "HJS",
                "UJ1",
                "UT1",
                "UV2",
                "UG1",
                "USA",
                "AAA",
                "ACA",
                "ACB",
                "AJS",
                "AHJ",
                "SPW",
            ): UndeterminedTimePeriod("2024-MAR-01"),
            ("BSL",): TimePeriod("2024-JUL-01", "2024-SEP-01"),
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
                "FJM",
                "FPM",
                "FSM",
                "FMR",
                "LBS",
                "LAN",
                "KEH",
                "AWWJ",
                "LAB",
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
            ("BCV",): UndeterminedTimePeriod("2024-JUN-01"),
            ("SWS", "XBM", "XBK", "XLB", "XPC"): UndeterminedTimePeriod(
                "2024-JAN-01"
            ),
        }
    )
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
        ("LBP", "LBT"): "LBS",
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


@dataclass
class InSeason:
    """Keeps track of which season labels are considered in season, based on
    refill type and dispatch date/month."""

    data: dict[Month, dict[RefillType, list[Season]]]
    """Inner datatype used to keep track of how in-season information."""

    def set_month(
        self, dispatch_month: Month, value: dict[RefillType, list[Season]]
    ):
        """Set the season labels considered in-season for a dispatch month."""
        self.data[dispatch_month] = value

    def get_in_season_for_month(
        self, dispatch_month: Month, refill_type: RefillType
    ) -> list[Season]:
        """Given the month the dispatch is being executed in, and the type of
        refill, get a list of the season labels considered in-season."""
        return self.data[dispatch_month][refill_type]

    def get_in_season_for_date(
        self, dispatch_date: DateLike, refill_type: RefillType
    ) -> pl.Series:
        """Given the date of dispatch, calculate which season labels for SKUs
        are in-season."""
        dispatch_date = Date.from_datelike(dispatch_date)
        return pl.Series(
            "season",
            [
                x.name
                for x in self.get_in_season_for_month(
                    dispatch_date.month, refill_type
                )
            ],
            dtype=Season.polars_type(),
        )

    @classmethod
    def from_off_season_info(cls, data: dict[Month, Season | None]) -> Self:
        """Generate in-season information based on off-season information."""
        result = cls({})
        for month, off_season in data.items():
            if off_season is None:
                off_seasons = []
            else:
                off_seasons = [off_season]

            result.set_month(
                month,
                dict(
                    (refill_type, refill_type.in_season()(off_seasons))
                    for refill_type in RefillType
                ),
            )
        return result


IN_SEASON_INFO: InSeason = InSeason.from_off_season_info(OFF_SEASONS)
"""Mapping containing information about what season labels are in-season, based
on the dispatch month and refill type being considered."""

# from historic_extract.ipynb
PER_CHANNEL_REFERENCE_CHANNELS: Mapping[
    Channel, Mapping[Category, Aggregator]
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

STRATEGY_LIBRARY: dict[StrategyId, list[InputStrategy]] = {
    LATEST: [
        InputStrategy(
            Channel.from_str("Amazon.com"),
            REFERENCE_CATEGORIES.as_dict(),
            defaultdict(
                lambda: UndeterminedTimePeriod("2024-FEB"),
                CURRENT_PERIODS.as_dict(),
            ),
            2023,
            PER_CHANNEL_REFERENCE_CHANNELS,
        ),
        InputStrategy(
            Channel.from_str("Amazon.ca"),
            REFERENCE_CATEGORIES.as_dict(),
            defaultdict(
                lambda: UndeterminedTimePeriod("2024-FEB"),
                CURRENT_PERIODS.as_dict(),
            ),
            2023,
            PER_CHANNEL_REFERENCE_CHANNELS,
        ),
    ],
}


def get_strategy_from_library(id: StrategyId) -> list[InputStrategy]:
    try:
        return STRATEGY_LIBRARY[id]
    except KeyError as e:
        raise KeyError(
            f"Strategy {id} not found. Available strategies are:"
            f" {sorted(STRATEGY_LIBRARY.keys())}"
        ) from e
