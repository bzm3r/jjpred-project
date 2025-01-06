"""An input strategy is a human-friendly form of configuring how prediction
information needs to be generated for each channel that requires a prediction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import auto
from functools import total_ordering
from typing import Any, Self, cast
import polars as pl
import datetime as dt

from jjpred.aggregator import Aggregator, UsingChannels
from jjpred.channel import Channel
from jjpred.globalvariables import (
    WEEKLY_PREDICTION_OFFSET,
    SEASON_START_PREDICTION_OFFSET,
)
from jjpred.seasons import Season
from jjpred.sku import Category


from jjpred.utils.datetime import (
    Date,
    DateLike,
    DateOffset,
    DateUnit,
    YearMonthDay,
    first_day_next_month,
    first_day,
    offset_date,
)
from jjpred.utils.polars import EnumLike
from jjpred.utils.pprint import PrettyPrint
from jjpred.utils.typ import (
    ScalarOrList,
    as_list,
    normalize_default_dict,
)


class RefillType(EnumLike):
    SEASON_START = auto()
    """Large refill: 3 months into the future prediction."""
    MONTHLY = auto()
    """Month-start refill: includes refill for some items that are only refilled
      monthly."""
    WEEKLY = auto()
    """Standard weekly refill."""
    CUSTOM_2024_SEP_10 = auto()
    """Custom refills should be created with a date so that they can be
    reproduced and used later."""
    CUSTOM_2024_NOV_04 = auto()
    CUSTOM_2025_JAN_06 = auto()

    @classmethod
    def match(cls, x: str) -> Self:
        x = x.lower()
        for variant in cls:
            if variant.name.lower() == x:
                return variant

        raise ValueError(f"Could not parse {x} as {cls}.")

    def end_date(self, start_date: DateLike) -> Date:
        """Get the end date of the prediction period for this refill type."""
        match self:
            case RefillType.MONTHLY | RefillType.WEEKLY:
                return offset_date(start_date, WEEKLY_PREDICTION_OFFSET)
            case RefillType.SEASON_START:
                return offset_date(start_date, SEASON_START_PREDICTION_OFFSET)
            case RefillType.CUSTOM_2024_SEP_10:
                start_date = Date.from_datelike(start_date)
                assert start_date == Date.from_datelike("2024-SEP-01")
                return Date.from_datelike("2024-DEC-01")
            case RefillType.CUSTOM_2024_NOV_04:
                start_date = Date.from_datelike(start_date)
                assert start_date == Date.from_datelike("2024-NOV-01")
                return Date.from_datelike("2025-JAN-15")
            case RefillType.CUSTOM_2025_JAN_06:
                start_date = Date.from_datelike(start_date)
                assert start_date == Date.from_datelike("2024-JAN-06")
                return Date.from_datelike("2025-APR-01")
            case value:
                raise ValueError(f"No logic to handle case {value}.")

    def in_season(
        self,
    ) -> Callable[[list[Season]], list[Season]]:
        """Get seasons considered "in-season" for this refill type."""
        match self:
            case (
                RefillType.MONTHLY
                | RefillType.SEASON_START
                | RefillType.CUSTOM_2024_SEP_10
                | RefillType.CUSTOM_2024_NOV_04
                | RefillType.CUSTOM_2025_JAN_06
            ):
                return lambda off_seasons: Season.all_seasons()
            case RefillType.WEEKLY:
                return lambda off_seasons: [
                    season
                    for season in Season
                    if ((season not in off_seasons) or (len(off_seasons) == 0))
                ]
            case value:
                raise ValueError(f"No logic to handle case {value}.")

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}.{self.name}"


@total_ordering
class UndeterminedTimePeriod:
    """Unknown end date. Usually determined later when a dispatch date is given."""

    start: Date

    def __init__(
        self,
        start: DateLike,
    ):
        self.start = Date.from_datelike(start)

    def with_end_date(self, end: DateLike) -> TimePeriod:
        return TimePeriod(self.start, end)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.start == other.start
        else:
            return False

    def __lt__(self, other: Self) -> bool:
        return self.start <= other.start


@dataclass
class HistoryTimePeriod:
    historical_year: TimePeriod
    """The historical year whose sales data we use to generate initial monthly
    ratios."""
    working_year: TimePeriod
    """The working year whose sales data we use to adjust historical year
    monthly ratios."""
    months_missing_from_working_year: list[int] = field(
        default_factory=list, init=False
    )
    """Incomplete months from the working year."""

    def __init__(self, historical_year: TimePeriod, working_year: TimePeriod):
        self.historical_year = historical_year
        self.working_year = working_year
        self.months_missing_from_working_year = [
            n
            for n in self.historical_year.tpoints.dt.month()
            if n not in self.working_year.tpoints.dt.month()
        ]


@total_ordering
class TimePeriod:
    start: Date
    end: Date
    tpoints: pl.Series
    """Timepoints making up this time period."""

    def __init__(
        self,
        start: DateLike,
        end: DateLike,
        tpoints: pl.Series | None = None,
    ):
        self.start = Date.from_datelike(start).with_day(1)
        self.end = Date.from_datelike(end).with_day(1)
        if tpoints is not None:
            assert tpoints[0] == start
            assert tpoints[-1] == first_day_next_month(self.end)
            self.tpoints = tpoints
        else:
            self.tpoints = pl.date_range(
                self.start.date,
                self.end.date,
                closed="left",
                eager=True,
                interval=str(DateOffset(1, DateUnit.MONTH)),
            )

    def with_end_date(self, end_date: DateLike) -> TimePeriod:
        end_date = Date.from_datelike(end_date)
        if self.end == end_date:
            return self
        else:
            return TimePeriod(self.start, end_date)

    @classmethod
    def make_current_period(
        cls,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
    ) -> Self:
        """A "current time period" is arbitrary in start/end date."""
        today: DateLike | None = None

        if start_date is None:
            today = Date.from_date(dt.date.today())
            start_date = Date.from_datelike(YearMonthDay(today.year, 1, 1))

        if end_date is None:
            if today is None:
                today = Date.from_date(dt.date.today())
            end_date = first_day(today)
        return cls(start_date, end_date)

    @classmethod
    def make_history_period(cls, dispatch_date: Date) -> HistoryTimePeriod:
        """A history time period has two parts: the historical year (12 time
        points) and the working year (most recent (completed) month in year of
        dispatch date)."""

        if dispatch_date.month == 1:
            working_year_start = YearMonthDay(dispatch_date.year - 1, 1, 1)
            working_year_end = YearMonthDay(dispatch_date.year, 1, 1)
        else:
            working_year_start = YearMonthDay(dispatch_date.year, 1, 1)
            working_year_end = YearMonthDay(
                dispatch_date.year, dispatch_date.month, 1
            )

        historical_year = dispatch_date.year - 1

        return HistoryTimePeriod(
            cls(
                Date.from_datelike(YearMonthDay(historical_year, 1, 1)),
                Date.from_datelike(YearMonthDay(historical_year + 1, 1, 1)),
            ),
            cls(working_year_start, working_year_end),
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.start == other.start and self.end == other.end
        else:
            return False

    def __lt__(self, other: Self) -> bool:
        return self.start <= other.start and self.end < other.end

    def __repr__(self) -> str:
        return self.tpoints.__repr__()


def fix_reference_chains(
    referred_to_primary_map: dict[Category, Category], raise_error: bool = True
) -> dict[Category, Category]:
    """Remove or detect any "reference chains" (where the referred to category
    itself refers to another category) in the referred category to primary
    category map."""
    indirections: dict[Category, list[Category]] = {}

    for cat in referred_to_primary_map.keys():
        ref_cats = []
        ref_cat = referred_to_primary_map.get(cat)
        while ref_cat is not None:
            ref_cats.append(ref_cat)
            ref_cat = referred_to_primary_map.get(ref_cat)

        indirections[cat] = ref_cats
        if len(ref_cats) == 0:
            raise ValueError(
                f"{cat} has no ref cat, yet it is a key in "
                f"{referred_to_primary_map=}"
            )

    problematic = {}
    for cat, ref_cats in indirections.items():
        if len(ref_cats) > 1:
            problematic[cat] = ref_cats

    if len(problematic) > 0:
        message = f"problematic ref chains: {problematic}"
        if raise_error:
            raise ValueError(f"problematic ref chains: {problematic}")
        else:
            print(f"WARNING: {message}")

    return dict([(k, v[-1]) for k, v in indirections.items()])


@dataclass
class InputStrategy(PrettyPrint):
    """A human-friendly definition of the strategy used to produce an FBA refill
    prediction."""

    channels: list[Channel]
    referred_to_primary_map: dict[Category, Category]
    current_periods: defaultdict[Category, TimePeriod | UndeterminedTimePeriod]
    aggregators: defaultdict[Category, Aggregator]

    def __pprint_items__(self) -> dict[str, Any]:
        return self.__dict__

    def __init__(
        self,
        channel: str | Channel,
        referred_to_primary_map: dict[Category, Category],
        current: TimePeriod
        | defaultdict[Category, TimePeriod | UndeterminedTimePeriod],
        per_channel_aggregator_map: Mapping[
            Channel, Aggregator | Mapping[Category, Aggregator]
        ],
    ):
        self.channels = [Channel.parse(x) for x in as_list(channel)]
        self.referred_to_primary_map = fix_reference_chains(
            referred_to_primary_map, raise_error=False
        )

        self.current_periods = normalize_default_dict(current)

        if not isinstance(current, defaultdict):
            self.current_periods = defaultdict(lambda: current, {})
        else:
            self.current_periods

        aggregators = per_channel_aggregator_map[Channel.parse(channel)]
        if not (
            isinstance(aggregators, defaultdict)
            or isinstance(aggregators, Aggregator)
        ):
            aggregators = dict(aggregators)
            self.aggregators = defaultdict(
                lambda: UsingChannels(
                    cast(ScalarOrList[Channel | str], self.channels)
                ),
                aggregators,
            )
        else:
            self.aggregators = normalize_default_dict(aggregators)
