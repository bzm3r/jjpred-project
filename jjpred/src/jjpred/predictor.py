"""Objects utilized to calculate the predicted demand in a given prediction
period."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Self
import datetime as dt

import polars as pl
import polars.selectors as cs

from jjpred.analysisdefn import FbaRevDefn, LatestDates, RefillDefn
from jjpred.channel import Channel
from jjpred.datagroups import (
    ALL_SKU_AND_CHANNEL_IDS,
    ALL_SKU_IDS,
    NOVELTY_FLAGS,
)
from jjpred.globalvariables import (
    LOW_CATEGORY_HISTORICAL_SALES,
    LOW_CURRENT_PERIOD_SALES,
    NEW_CATEGORIES,
    OUTPERFORM_FACTOR,
)
from jjpred.inputstrategy import TimePeriod
from jjpred.performanceflags import PerformanceFlag
from jjpred.predictiontypes import PredictionType
from jjpred.readsupport.utils import cast_standard
from jjpred.readsupport.predictiontypes import read_prediction_types
from jjpred.skuinfo import get_all_sku_currentness_info
from jjpred.utils.datamanipulation import merge_wholesale
from jjpred.utils.groupeddata import (
    CategoryGroupProtocol,
    CategoryGroups,
    ChannelCategoryData,
)
from jjpred.strategies import (
    CategoryType,
    CategoryGroup,
    ChannelStrategyGroups,
    StrategyGroup,
    collate_groups_per_channel,
)
from jjpred.database import DataBase
from jjpred.sku import Category, Sku
from jjpred.structlike import MemberType
from jjpred.utils.datetime import (
    Date,
    DateLike,
    first_day_next_month,
    first_day,
)
from jjpred.utils.multidict import MultiDict
from jjpred.utils.polars import (
    NoOverride,
    OverrideLeft,
    polars_float,
    vstack_to_unified,
    find_dupes,
    join_and_coalesce,
)

from jjpred.utils.typ import (
    ScalarOrList,
    normalize_as_list,
)
from jjpred.readsheet import DataVariant


@dataclass
class MonthPart:
    """A month-part is some portion of a month; possibly the whole month. The
    end date is always not included in period. For example: a month-part
    starting at 2024-DEC-01 and ending at 2025-JAN-01 includes all the days in
    December not including the first day of January."""

    start: Date
    """Start of the month-part."""
    end: Date
    """End of the month-part. Must be greater than the start date, and must
    either be some date in the same month as the start date, or the first day of
   the next month."""

    def __init__(self, start: Date, end: Date):
        assert (start <= end) and (
            (start.month == end.month)
            or (
                (
                    (start.month + 1) == end.month
                    or (start.month == 12 and end.month == 1)
                )
                and end.day == 1
            )
        ), (start, end)
        self.start = start
        self.end = end


@dataclass
class PeriodBreakdown:
    """Breakdown a period across months, into month-parts."""

    start_date: Date
    """Start of the period."""
    end_date: Date
    """End of the period."""
    sub_periods: list[MonthPart]
    """The month-parts that this period is divided into."""
    df: pl.DataFrame
    """Dataframe containing all relevant information of this period's
    breakdown."""

    @classmethod
    def make_dates_series(cls, dates: list[dt.date]) -> pl.Series:
        """Create a Polars series named "date" from the given dates."""
        return pl.Series("date", dates, dtype=pl.Date())

    @classmethod
    def make_days_from_month_series(
        cls, days_from_month: list[int]
    ) -> pl.Series:
        """Make a Polars series named "days_from_month" from a list of
        integers."""
        return pl.Series("days_from_month", days_from_month, dtype=pl.Int8())

    @classmethod
    def make_days_in_month_series(cls, days_in_month: list[int]) -> pl.Series:
        return pl.Series("days_in_month", days_in_month, dtype=pl.Int8())

    @classmethod
    def __simple__(cls, start_date: Date, end_date: Date) -> Self:
        periods = [MonthPart(start_date, end_date)]

        if start_date == end_date:
            dates = []
            days_from_month = []
            days_in_month = []
        else:
            dates = [start_date.date]
            days_from_month = [(end_date - start_date).days]
            if (
                start_date.month == end_date.month
                and start_date.year == end_date.year
            ):
                next_tpoint = first_day_next_month(end_date)
            else:
                next_tpoint = first_day(end_date)
            days_in_month = [(next_tpoint - first_day(start_date)).days]

        df = pl.DataFrame(
            [
                cls.make_dates_series(dates),
                cls.make_days_from_month_series(days_from_month),
                cls.make_days_in_month_series(days_in_month),
            ]
        ).with_columns(
            month=pl.col("date").dt.month(),
            month_fraction=pl.col.days_from_month / pl.col.days_in_month,
        )

        return cls(
            start_date,
            end_date,
            periods,
            df,
        )

    def __vstack__(self, other: Self) -> Self:
        assert self.end_date == other.start_date
        df = self.df.vstack(other.df.select(self.df.columns))
        return self.__class__(
            self.start_date,
            other.end_date,
            self.sub_periods + other.sub_periods,
            df,
        )

    def month_fractions(
        self, match_main_program_month_fractions: bool = False
    ) -> pl.DataFrame:
        """For each month-part making up the prediction, get what fraction of
        the month the month-part is.

        For example, a month-part of 15 days will be 50% of of a 30 day month,
        but slightly more than 53.6% for a 28 day month.

        If ``match_main_program_month_fractions`` is ``True``, then round the month fractions to
        a multiple of 25%."""

        month_fractions = self.df.with_columns(
            month=pl.col("date").dt.month()
        ).select("date", "month", "month_fraction")

        if match_main_program_month_fractions:
            month_fractions = (
                month_fractions.with_columns(
                    num_month_parts=(pl.col.month_fraction / 0.25).round()
                )
                .with_columns(month_fraction=pl.col.num_month_parts * 0.25)
                .drop("num_month_parts")
            )

        return month_fractions

    @classmethod
    def generate_breakdown(
        cls,
        start_date: DateLike,
        end_date: DateLike,
    ) -> Self:
        """Generate a breakdown for a period defined by the given start and
        end date."""
        start_date = Date.from_datelike(start_date)
        end_date = Date.from_datelike(end_date)

        if start_date > end_date:
            raise ValueError(f"{start_date} > {end_date}")
        elif (start_date == end_date) or (
            start_date.month == end_date.month
            and start_date.year == end_date.year
        ):
            return cls.__simple__(
                start_date,
                end_date,
            )
        else:
            next_start = first_day_next_month(start_date)
            this = cls.__simple__(
                start_date,
                next_start,
            )
            remainder = cls.generate_breakdown(next_start, end_date)
            return this.__vstack__(remainder)


class POPrediction:
    """Object for managing prediction information based on PO data."""

    monthly_expected_demand_from_po: pl.DataFrame
    """Expected monthly demand based on PO data"""

    def __init__(self, strategy: StrategyGroup, po_data: pl.DataFrame):
        self.monthly_expected_demand_from_po = (
            strategy.construct_po_prediction(
                po_data,
            )
        )

    def expected_demand_in_period(
        self,
        start_date: DateLike,
        end_date: DateLike,
        match_main_program_month_fractions: bool = False,
    ) -> pl.DataFrame:
        """Calculate the expected demand in a period based on the PO data.

        We divide the monthly estimate from PO data into a demand per day. Then,
        we multiply the number of days from each month making up the period by
        this demand per day to get the demand per month."""
        start_date = Date.from_datelike(start_date)
        end_date = Date.from_datelike(end_date)
        prediction_month_fractions = PeriodBreakdown.generate_breakdown(
            start_date, end_date
        ).month_fractions(
            match_main_program_month_fractions=match_main_program_month_fractions
        )

        expected_demand = (
            self.monthly_expected_demand_from_po.join(
                prediction_month_fractions,
                on="month",
                # there are multiple entries in the LHS with the same month
                validate="m:1",
                join_nulls=True,
            ).with_columns(
                expected_demand_from_po=(
                    pl.col("monthly_expected_demand_from_po")
                    * pl.col.month_fraction
                )
                .ceil()
                .cast(pl.Int64())
            )
            # .group_by(
            #     cs.exclude(
            #         "month",
            #         "month_fraction",
            #         "monthly_expected_demand_from_po",
            #         "date",
            #         "expected_demand_from_po",
            #     )
            # )
            # .agg(pl.col("expected_demand_from_po").sum())
        )

        return expected_demand.select(
            ["sku", "a_sku"]
            + Channel.members()
            + ["date"]
            + ["monthly_expected_demand_from_po", "expected_demand_from_po"]
        )


class CurrentPeriodSales:
    """Manages current period sales information."""

    monthly_sales: pl.DataFrame
    """Monthly sales in the current period."""
    total_sales: pl.DataFrame
    """Total salesin the current period."""
    tpoints: pl.Series
    """Time points making up the current period."""

    @classmethod
    def from_strategy(
        cls,
        strategy: StrategyGroup,
        history_df: pl.DataFrame,
        default_current_period_end_date: DateLike,
    ) -> MultiDict[Category, Self]:
        all_monthly_sales = strategy.construct_current(
            default_current_period_end_date, history_df
        )
        result = MultiDict({})
        for categories, monthly_sales in all_monthly_sales.data.items():
            current_period = strategy.current_periods.data[categories]
            assert isinstance(current_period, TimePeriod)
            result.data[categories] = cls(current_period, monthly_sales)

        return result

    def __init__(
        self,
        current_period: TimePeriod,
        monthly_sales: pl.DataFrame,
    ):
        self.monthly_sales = monthly_sales
        self.total_sales = self.monthly_sales.group_by(
            cs.exclude("current_monthly_sales", "date")
        ).agg(
            pl.col("current_monthly_sales").sum().alias("current_period_sales")
        )
        self.tpoints = current_period.tpoints


class ExpectedYearSales:
    """Manages information about the total expected year sales for SKUs."""

    expected_year_sales_full: pl.DataFrame
    """Expected year sales with full details."""

    @classmethod
    def from_current_period_sales(
        cls,
        historical: HistoricalPeriodSales,
        currents: MultiDict[Category, CurrentPeriodSales],
    ) -> MultiDict[Category, Self]:
        result = MultiDict({})
        for categories, current in currents.data.items():
            result.data[categories] = cls(historical, current)
        return result

    def __init__(
        self, historical: HistoricalPeriodSales, current: CurrentPeriodSales
    ):
        current_demand_ratio = (
            historical.demand_ratios.filter(
                pl.col("month").is_in(current.tpoints.dt.month())
            )
            .group_by("category")
            .agg(current_demand_ratio=pl.col("demand_ratio").sum())
        )

        self.expected_year_sales_full = current.total_sales.join(
            current_demand_ratio,
            on="category",
            # each LHS category has print, size, etc.
            # each RHS category has month, current
            validate="m:m",
            join_nulls=True,
        ).with_columns(
            pl.when(pl.col("current_demand_ratio") > 0.0)
            .then(
                (
                    pl.col("current_period_sales")
                    / pl.col("current_demand_ratio")
                )
            )
            .otherwise(pl.lit(0.0))
            .alias("expected_year_sales")
        )

        assert (
            len(
                find_dupes(
                    self.expected_year_sales_full,
                    list(
                        cs.expand_selector(
                            self.expected_year_sales_full,
                            cs.exclude("expected_year_sales"),
                        )
                    ),
                )
            )
            == 0
        )

    @property
    def expected_year_sales(self) -> pl.DataFrame:
        """Expected year sales information."""
        return self.expected_year_sales_full


@dataclass
class HistoricalPeriodSales:
    """Manages information about the historical sales in the historical period.
    This information is provided at the category level."""

    category_historical_year_sales_info: pl.DataFrame = field(init=False)
    """Sales in the historical year, per category."""
    category_working_year_sales_info: pl.DataFrame = field(init=False)
    """Sales in the working year, per category."""
    category_sales_info: pl.DataFrame = field(init=False)
    """Monthly demand ratios, based on working year sales info and adjusted
    against historical year sales info."""

    def __init__(
        self,
        strategy: StrategyGroup,
        sales_history_df: pl.DataFrame,
        demand_ratio_rolling_update_to_date: Date,
    ):
        history_dfs = strategy.construct_history(
            sales_history_df, demand_ratio_rolling_update_to_date
        )

        self.category_historical_year_sales_info = (
            history_dfs.historical_df.with_columns(
                category_historical_year_sales=pl.col(
                    "category_historical_monthly_sales"
                )
                .sum()
                .over("category")
            )
        )

        self.category_working_year_sales_info = (
            history_dfs.working_df.with_columns(
                category_working_year_sales=pl.col(
                    "category_working_monthly_sales"
                )
                .sum()
                .over("category")
            ).with_columns(month=pl.col.date.dt.month())
        )

        self.category_historical_year_sales_info = (
            self.category_historical_year_sales_info.with_columns(
                historical_demand_ratio=pl.when(
                    pl.col("category_historical_year_sales") > 0
                )
                .then(
                    pl.col("category_historical_monthly_sales")
                    / pl.col("category_historical_year_sales")
                )
                .otherwise(pl.lit(-1.0))
                .cast(polars_float(64))
            ).with_columns(month=pl.col("date").dt.month())
        )

        self.category_sales_info = (
            self.category_historical_year_sales_info.select(
                "category",
                "month",
                "historical_demand_ratio",
                "category_historical_monthly_sales",
                "category_historical_year_sales",
            )
        )

        working_months_historical_ratio_df = (
            self.category_sales_info.filter(
                pl.col.month.is_in(
                    self.category_working_year_sales_info["month"]
                    .unique()
                    .sort()
                )
            )
            .group_by("category")
            .agg(
                pl.col.historical_demand_ratio.sum().alias(
                    "working_months_historical_ratio"
                )
            )
        )

        self.category_working_year_sales_info = (
            self.category_working_year_sales_info.join(
                working_months_historical_ratio_df.select(
                    "category", "working_months_historical_ratio"
                ).unique(),
                on=["category"],
                validate="m:1",
            ).with_columns(
                working_demand_ratio=pl.when(
                    pl.col.category_working_year_sales.gt(0)
                )
                .then(
                    (
                        pl.col.category_working_monthly_sales
                        / pl.col.category_working_year_sales
                    )
                    * pl.col.working_months_historical_ratio
                )
                .otherwise(pl.lit(-1.0))
            )
        )

        self.category_sales_info = self.category_sales_info.join(
            self.category_working_year_sales_info.select(
                "category",
                "month",
                "category_working_monthly_sales",
                "category_working_year_sales",
                "working_demand_ratio",
                "working_months_historical_ratio",
            ),
            on=["category", "month"],
            how="left",
            validate="1:1",
        ).with_columns(
            demand_ratio=pl.when(~pl.col.historical_demand_ratio.lt(0.0))
            .then(
                pl.when(~pl.col.working_demand_ratio.lt(0.0))
                .then(pl.col.working_demand_ratio)
                .otherwise(pl.col.historical_demand_ratio)
            )
            .otherwise(pl.lit(-1.0))
        )

        self.category_sales_info = self.category_sales_info

    @property
    def demand_ratios(self) -> pl.DataFrame:
        """Get the monthly demand ratios based on historical sales
        information."""
        return self.category_sales_info.select(
            cs.exclude(
                "category_historical_monthly_sales",
                "category_historical_year_sales",
                "days_in_month",
            )
        )

    @property
    def category_year_sales(self):
        """The yearly sales information this category."""
        return self.category_sales_info.select(
            "category", "category_historical_year_sales"
        ).unique()

    def prediction_demand_ratios(
        self,
        start_date: DateLike,
        end_date: DateLike,
        match_main_program_month_fractions: bool = False,
    ) -> pl.DataFrame:
        """Calculate the demand ratios for each month-part of the prediction
        period."""
        start_date = Date.from_datelike(start_date)
        end_date = Date.from_datelike(end_date)

        prediction_period_month_fractions = PeriodBreakdown.generate_breakdown(
            start_date,
            end_date,
        ).month_fractions(match_main_program_month_fractions)

        demand_ratios = self.demand_ratios.join(
            prediction_period_month_fractions,
            on="month",
            # there are multiple categories in the LHS
            validate="m:1",
            join_nulls=True,
        )

        return demand_ratios

    def total_prediction_demand_ratio(
        self, start_date: DateLike, end_date: DateLike
    ) -> pl.DataFrame:
        """Calculate the total demand ratio for the entire prediction period."""
        return (
            self.prediction_demand_ratios(start_date, end_date)
            .select(
                "category",
                "date",
                "month_fraction",
                "demand_ratio",
            )
            .with_columns(
                demand_ratio=pl.col.month_fraction * pl.col.demand_ratio
            )
            .drop("month_fraction")
            .group_by("category")
            .agg(pl.col("demand_ration").sum())
        )


class PredictionInput(CategoryGroupProtocol):
    strategy: StrategyGroup
    historical: HistoricalPeriodSales
    currents: MultiDict[Category, CurrentPeriodSales]
    po_prediction: POPrediction
    expected_year_sales: MultiDict[Category, ExpectedYearSales]
    use_po_categories: dict[PredictionType, dict[CategoryType, list[Category]]]

    @property
    def all_categories(self) -> list[Category]:
        return self.strategy.all_categories

    def expected_year_sales_full(self) -> pl.DataFrame:
        """Expected year sales (across all category groups)."""
        result = None
        for _, eys in self.expected_year_sales.data.items():
            result = vstack_to_unified(result, eys.expected_year_sales)

        assert isinstance(result, pl.DataFrame)
        return result

    def __init__(
        self,
        strategy: StrategyGroup,
        sales_history_df: pl.DataFrame,
        po_data: pl.DataFrame,
        latest_dates: LatestDates,
    ):
        self.strategy = strategy
        self.historical = HistoricalPeriodSales(
            strategy,
            sales_history_df,
            latest_dates.demand_ratio_rolling_update_to,
        )
        self.po_prediction = POPrediction(strategy, po_data)
        self.currents = CurrentPeriodSales.from_strategy(
            strategy, sales_history_df, latest_dates.sales_history_latest_date
        )
        self.expected_year_sales = ExpectedYearSales.from_current_period_sales(
            self.historical, self.currents
        )

    def expected_demand_from_history(
        self,
        all_sku_info: pl.DataFrame,
        start_date: DateLike,
        end_date: DateLike,
        filter_a_sku: pl.Series | None = None,
        aggregate: bool = False,
        match_main_program_month_fractions: bool = False,
    ) -> pl.DataFrame:
        pd = self.historical.prediction_demand_ratios(
            start_date,
            end_date,
            match_main_program_month_fractions=match_main_program_month_fractions,
        ).select(
            "category",
            "date",
            "month_fraction",
            "demand_ratio",
        )
        expected_year_sales = self.expected_year_sales_full().select(
            ["a_sku", "category"] + Channel.members() + ["expected_year_sales"]
        )
        if filter_a_sku is not None:
            expected_year_sales = expected_year_sales.filter(
                pl.col.a_sku.is_in(filter_a_sku)
            )
        find_dupes(
            expected_year_sales,
            id_cols=["a_sku", "category"] + Channel.members(),
            raise_error=True,
        )
        expected_demand_from_history = (
            expected_year_sales.join(
                pd,
                on="category",
                # each category in LHS has print, size etc.
                # each cateogry in RHS has prediction ratios per month
                validate="m:m",
                join_nulls=True,
            ).with_columns(
                monthly_expected_demand_from_history=pl.col.demand_ratio
                * pl.col.expected_year_sales,
                expected_demand_from_history=(
                    pl.col.month_fraction
                    * pl.col.demand_ratio
                    * pl.col.expected_year_sales
                )
                .ceil()
                .cast(pl.Int64()),
            )
            # .group_by(["a_sku", "category"] + Channel.members())
            # .agg(pl.col("expected_demand_from_history").sum())
        )

        remaining_info = [
            "monthly_expected_demand_from_history",
            "expected_demand_from_history",
        ]
        if aggregate:
            remaining_info = ["expected_demand_from_history"]

        expected_demand_from_history = (
            (
                expected_demand_from_history.join(
                    all_sku_info.select("sku", "a_sku"),
                    on="a_sku",
                )
            )
            .drop("category")
            .select(
                ["a_sku", "sku"]
                + Channel.members()
                + ["date"]
                + remaining_info
            )
        )

        if aggregate:
            expected_demand_from_history = (
                expected_demand_from_history.group_by(
                    ["a_sku", "sku"] + Channel.members()
                )
            ).agg(pl.col.expected_demand_from_history.sum())

        return expected_demand_from_history


class PredictionInputs(CategoryGroups[PredictionInput]):
    pass


class Predictor(ChannelCategoryData[PredictionInputs, PredictionInput]):
    db: DataBase
    """Database used to generate this predictor."""
    # in_season_skus: pl.DataFrame = field(init=False)
    # """Which SKUs are being considered in-season?"""
    # out_season_skus: pl.DataFrame = field(init=False)
    # """Which SKUs are being considered out-(of)-season?"""
    prediction_type_meta_date: str | DateLike | None = None
    """Meta-information marking the particular prediction types input file to use."""
    prediction_types: pl.DataFrame
    """Prediction types dataframe generated from the prediction types input
    file."""
    analysis_defn: RefillDefn
    """Analysis definition governing this predictor instance."""
    strategy_groups_per_channel: ChannelStrategyGroups
    """Strategy groups defining how prediction is to be done per channel."""
    history_data: pl.DataFrame
    """Dataframe containing historical sales information."""
    po_data: pl.DataFrame
    """Dataframe containing expected demands based on the PO process."""
    latest_dates: LatestDates
    """The latest dates used to calculate current period sales and latest demand
    (monthly) ratios."""
    data: dict[Channel, PredictionInputs] = {}
    """Data used to generate predictions, per channel."""
    merge_faire_and_wholesale_into_wholesale: bool = True
    """Whether to merge Faire.com and Wholesale data into the Wholesale channel."""

    def __init__(
        self,
        analysis_defn: RefillDefn,
        db: DataBase,
        strategy_groups_per_channel: ChannelStrategyGroups,
        po_data: pl.DataFrame,
        latest_dates: LatestDates | None = None,
        merge_faire_and_wholesale_into_wholesale: bool = True,
    ) -> None:
        self.db = db
        self.analysis_defn = analysis_defn
        self.prediction_type_meta_date = (
            analysis_defn.prediction_type_meta_date
        )
        self.prediction_types = read_prediction_types(
            self.db, self.prediction_type_meta_date
        )

        self.history_data = self.db.dfs[DataVariant.History]

        if merge_faire_and_wholesale_into_wholesale:
            self.history_data = merge_wholesale(
                self.history_data,
                ["a_sku", "date", "category"],
                [pl.col.sales.sum()],
            )

        self.history_data = self.history_data.drop("channel")
        find_dupes(
            self.history_data,
            ["a_sku", "date", "category"] + Channel.members(),
            raise_error=True,
        )

        self.po_data = po_data

        self.strategy_groups_per_channel = strategy_groups_per_channel
        if latest_dates is None:
            self.latest_dates = self.analysis_defn.latest_dates
        else:
            self.latest_dates = latest_dates

        self.data = {}
        for (
            channel,
            strategy_groups,
        ) in self.strategy_groups_per_channel.items():
            prediction_inputs = []
            for group in strategy_groups.category_groups():
                prediction_inputs.append(
                    PredictionInput(
                        group,
                        self.history_data,
                        self.po_data,
                        self.latest_dates,
                    )
                )

            self.set_category_group_for_channel(
                channel, PredictionInputs(prediction_inputs)
            )

    def generate_data_using_latest_dates(
        self, latest_dates: LatestDates | None = None
    ) -> Predictor:
        """Create a predictor that uses current period sales and latest demand
        (monthly) ratios based on an optional given date.

        :param latest_dates: The latest dates used to calculate current period
            sales and latest demand (monthly) ratios, defaults to None. If None,
            it uses the latest available date.
        """
        return Predictor(
            self.analysis_defn,
            self.db,
            self.strategy_groups_per_channel,
            self.po_data,
            latest_dates,
        )

    def get_historical_sales(
        self,
        channels: ScalarOrList[str] | ScalarOrList[Channel],
    ) -> dict[Channel, CategoryGroups[CategoryGroup[pl.DataFrame]]]:
        """Get all historical period sales across all channels and SKUs."""
        channels = [Channel.parse(ch) for ch in normalize_as_list(channels)]
        historical_sales_per_channel = {}

        for channel in channels:
            historical_sales_per_group: list[CategoryGroup[pl.DataFrame]] = []
            for pd_inputs in self.get_category_groups_for_channel(
                channel
            ).category_groups():
                historical_sales_per_group.append(
                    CategoryGroup(
                        pd_inputs.strategy,
                        pd_inputs.historical.category_sales_info,
                    )
                )

            historical_sales_per_channel[channel] = CategoryGroups(
                historical_sales_per_group
            )

        return historical_sales_per_channel

    def get_current_period_monthly_sales(
        self,
        channels: ScalarOrList[str] | ScalarOrList[Channel],
    ) -> dict[Channel, CategoryGroups[CategoryGroup[pl.DataFrame]]]:
        """Get all current period sales across all channels and SKUs."""
        channels = [Channel.parse(ch) for ch in normalize_as_list(channels)]
        current_period_sales_per_channel = {}

        for channel in channels:
            current_period_sales_per_group: list[
                CategoryGroup[pl.DataFrame]
            ] = []
            for pd_inputs in self.get_category_groups_for_channel(
                channel
            ).category_groups():
                monthly_sales_df: None | pl.DataFrame = None
                for (
                    _,
                    current,
                ) in pd_inputs.currents.data.items():
                    monthly_sales_df = vstack_to_unified(
                        monthly_sales_df, current.monthly_sales
                    )

                if monthly_sales_df is not None:
                    current_period_sales_per_group.append(
                        CategoryGroup(pd_inputs.strategy, monthly_sales_df)
                    )

            current_period_sales_per_channel[channel] = CategoryGroups(
                current_period_sales_per_group
            )

        return current_period_sales_per_channel

    def get_current_total_sales(
        self,
        channels: ScalarOrList[str] | ScalarOrList[Channel],
    ) -> dict[Channel, CategoryGroups[CategoryGroup[pl.DataFrame]]]:
        """Get all current period total sales across all channels and SKUs."""
        channels = [Channel.parse(ch) for ch in normalize_as_list(channels)]
        current_total_sales_per_channel = {}

        for channel in channels:
            current_total_sales_per_group: list[
                CategoryGroup[pl.DataFrame]
            ] = []
            for pd_inputs in self.get_category_groups_for_channel(
                channel
            ).category_groups():
                current_total_sales: None | pl.DataFrame = None

                for _, current in pd_inputs.currents.data.items():
                    current_total_sales = vstack_to_unified(
                        current_total_sales, current.total_sales
                    )

                if current_total_sales is not None:
                    current_total_sales_per_group.append(
                        CategoryGroup(pd_inputs.strategy, current_total_sales)
                    )

            current_total_sales_per_channel[channel] = CategoryGroups(
                current_total_sales_per_group
            )

        return current_total_sales_per_channel

    def get_po_data(
        self,
        channels: ScalarOrList[str] | ScalarOrList[Channel],
    ) -> dict[Channel, CategoryGroups[CategoryGroup[pl.DataFrame]]]:
        """Get all PO data across all channels and SKUs."""
        channels = [Channel.parse(ch) for ch in normalize_as_list(channels)]
        po_data_per_channel = {}

        for channel in channels:
            po_data_per_group: list[CategoryGroup[pl.DataFrame]] = []
            for pd_inputs in self.get_category_groups_for_channel(
                channel
            ).category_groups():
                po_data_per_group.append(
                    CategoryGroup(
                        pd_inputs.strategy,
                        pd_inputs.po_prediction.monthly_expected_demand_from_po,
                    )
                )

            po_data_per_channel[channel] = CategoryGroups(po_data_per_group)

        return po_data_per_channel

    def get_current_sales_info(self) -> pl.DataFrame:
        """Get all current period sales information across all channels and
        SKUs."""
        channels = list(self.data.keys())
        all_current_period_sales_info = (
            collate_groups_per_channel(
                self.get_current_period_monthly_sales(channels)
            )
            .select(
                ["a_sku"]
                + Channel.members()
                + ["date", "current_monthly_sales"]
            )
            .group_by(cs.exclude("date", "current_monthly_sales"))
            .agg(
                pl.col("date"),
                pl.col("current_monthly_sales")
                .sum()
                .alias("current_period_sales"),
            )
            .with_columns(pl.col("date").list.sort())
            .with_columns(
                pl.col("date").list.first().alias("current_period_start"),
                pl.col("date")
                .list.last()
                .alias("current_period_end")
                .dt.month_start()
                .dt.offset_by("1mo"),
            )
            .drop("date")
            .with_columns(
                current_period=pl.concat_list(
                    "current_period_start", "current_period_end"
                )
            )
            .drop("current_period_start", "current_period_end")
        )

        return all_current_period_sales_info.unique()

    def get_historical_sales_info(
        self, channel_info: pl.DataFrame
    ) -> pl.DataFrame:
        """Get all historical period sales information across all channels and
        SKUs."""
        channels = list(self.data.keys())
        historical_sales_info = (
            cast_standard(
                [channel_info],
                collate_groups_per_channel(
                    self.get_historical_sales(channels),
                    attach_channel=True,
                ),
            )
            .select(
                ["category"]
                + Channel.members()
                + ["month", "category_historical_monthly_sales"]
            )
            .group_by(cs.exclude("month", "category_historical_monthly_sales"))
            .agg(
                pl.col("month"),
                pl.col("category_historical_monthly_sales")
                .sum()
                .alias("category_historical_year_sales"),
            )
            .with_columns(pl.col("month").list.sort())
            # .with_columns(
            #     pl.col("date").list.first().alias("historical_period_start"),
            #     pl.col("date")
            #     .list.last()
            #     .alias("historical_period_end")
            #     .dt.month_start()
            #     .dt.offset_by("1mo"),
            # )
            # .drop("date")
            # .with_columns(
            #     historical_period=pl.concat_list(
            #         "historical_period_start", "historical_period_end"
            #     )
            # )
            # .drop("historical_period_start", "historical_period_end")
        )

        return historical_sales_info

    def get_po_info(self) -> pl.DataFrame:
        """Get all PO data information across all channels and SKUs."""
        channels = list(self.data.keys())
        po_info = (
            collate_groups_per_channel(self.get_po_data(channels))
            .select(
                ALL_SKU_AND_CHANNEL_IDS + ["monthly_expected_demand_from_po"]
            )
            .group_by(cs.exclude("monthly_expected_demand_from_po"))
            .agg(
                pl.col("monthly_expected_demand_from_po")
                .sum()
                .alias("po_expected_total_demand")
            )
        )

        return po_info

    def get_category_types(self, db: DataBase) -> pl.DataFrame:
        """Get category types (whether primary or referred) across all channels
        and SKUs."""
        category_type_df = pl.DataFrame()

        for ch, gs in self.data.items():
            for g in gs.category_groups():
                category_types = g.strategy.category_types
                for category_type, categories in [
                    (k, v)
                    for k, v in category_types.items()
                    if k != CategoryType.ALL
                ]:
                    result_df = (
                        pl.DataFrame(pl.Series("category", categories))
                        .with_columns(**ch.to_columns())
                        .with_columns(category_type=pl.lit(category_type.name))
                        .with_columns(
                            refers_to=pl.col("category").map_elements(
                                lambda x: g.strategy.get_reference_category(x),
                                return_dtype=pl.String(),
                                skip_nulls=False,
                            ),
                        )
                        .with_columns(
                            referred_by=pl.col("category").map_elements(
                                lambda x: g.strategy.primary_to_referred.get(
                                    x, []
                                ),
                                return_dtype=pl.List(pl.String()),
                            ),
                        )
                    )
                    category_type_df = vstack_to_unified(
                        category_type_df, result_df
                    )

        category_type_df = cast_standard(
            [db.meta_info.active_sku, db.meta_info.channel],
            category_type_df,
            use_dtype_of={"refers_to": "category"},
        ).cast(
            {
                "referred_by": pl.List(
                    db.meta_info.active_sku["category"].dtype
                ),
                "category_type": pl.Enum(
                    category_type_df["category_type"]
                    .drop_nulls()
                    .unique()
                    .sort()
                ),
            },
        )

        return category_type_df

    def get_input_data_info(
        self, force_po_prediction: bool = False
    ) -> pl.DataFrame:
        """Get dataframe recording:
        * availability of historical sales data, current sales data, PO data
        * what kind of prediction type is used by each SKU
        * whether a category is new
        * whether a print is new
        * whether a category refers to another category for prediction
        * whether a category is marked "new" by marketing
        * etc.
        """

        current_sales_info = self.get_current_sales_info()

        historical_sales_info = self.get_historical_sales_info(
            self.db.meta_info.channel
        )

        po_info = self.get_po_info()

        category_type_df = self.get_category_types(self.db)

        sku_info_df = get_all_sku_currentness_info(self.db).filter(
            pl.col.is_active
        )

        latest_date = self.latest_dates.latest()
        print(f"{latest_date=}")
        joined_df = (
            sku_info_df.join(
                current_sales_info.select(Channel.members()).unique(),
                how="cross",
                on=None,
            )
            .join(
                current_sales_info,
                on=["a_sku"] + Channel.members(),
                how="left",
            )
            .join(
                historical_sales_info,
                on=["category"] + Channel.members(),
                how="left",
                coalesce=True,
                join_nulls=True,
            )
            .join(
                po_info,
                on=ALL_SKU_AND_CHANNEL_IDS,
                how="left",
                coalesce=True,
                join_nulls=True,
            )
            .join(
                self.prediction_types.filter(
                    pl.col.dispatch_month.eq(latest_date.month)
                )
                .drop("dispatch_month")
                .with_columns(
                    pl.when(pl.lit(force_po_prediction))
                    .then(
                        pl.lit(
                            "PO",
                            dtype=self.prediction_types[
                                "prediction_type"
                            ].dtype,
                        )
                    )
                    .otherwise(pl.col.prediction_type)
                    .alias("prediction_type")
                ),
                on=ALL_SKU_IDS,
                how="left",
                coalesce=True,
                join_nulls=True,
            )
            .join(
                category_type_df,
                on=["category"] + Channel.members(),
                how="left",
                coalesce=True,
                join_nulls=True,
            )
        )
        find_dupes(joined_df, ALL_SKU_AND_CHANNEL_IDS, raise_error=True)

        input_data_info = (
            joined_df.with_columns(
                has_current_data=(
                    pl.col.current_period_sales.is_not_null()
                    & pl.col.current_period_sales.gt(0)
                ),
                has_historical_data=(
                    pl.col.category_historical_year_sales.is_not_null()
                    & pl.col.category_historical_year_sales.gt(0)
                ),
                has_po_data=(
                    pl.col.po_expected_total_demand.is_not_null()
                    & pl.col.po_expected_total_demand.gt(0)
                ),
            )
            .select(
                ["a_sku"]
                + Sku.members(MemberType.META)
                + Channel.members()
                + NOVELTY_FLAGS
                + [
                    "current_period_sales",
                    "current_period",
                    "category_historical_year_sales",
                    "po_expected_total_demand",
                    "category_type",
                    "refers_to",
                    "referred_by",
                    "prediction_type",
                    "has_current_data",
                    "has_historical_data",
                    "has_po_data",
                ]
            )
            .with_columns(
                new_overrides_e=pl.lit(
                    self.analysis_defn.new_overrides_e, dtype=pl.Boolean()
                ),
                category_marked_new=pl.col("category").is_in(NEW_CATEGORIES),
            )
            .with_columns(
                has_e_data=pl.col("has_historical_data")
                & pl.col("has_current_data"),
            )
            .with_columns(
                uses_ce=pl.col.prediction_type.eq("CE"),
                uses_e=pl.col.prediction_type.eq("E"),
                uses_po=pl.col.prediction_type.eq("PO"),
            )
            .with_columns(
                # uses "NEW" mode prediction
                uses_ne=(
                    (pl.col.uses_po | pl.col.uses_ce)
                    & pl.col.category_marked_new
                )
                | (
                    pl.col.uses_e
                    & pl.col.new_overrides_e
                    & pl.col.category_marked_new
                )
            )
            .with_columns(
                # if we initially marked a category as using CE, but it is
                # marked as a "NEW" category, then unmark it as using CE,
                # because really it is using NE
                uses_ce=pl.when(pl.col.uses_ne)
                .then(pl.lit(False))
                .otherwise(pl.col.uses_ce)
            )
            .with_columns(
                low_current_period_sales=pl.when(
                    pl.col.has_current_data
                    & (
                        pl.col.uses_e
                        | pl.col.uses_ce
                        | (pl.col.uses_po & pl.col.category_marked_new)
                    )
                )
                .then(pl.col.current_period_sales.lt(LOW_CURRENT_PERIOD_SALES))
                .otherwise(True)
            )
            .with_columns(
                low_category_historical_sales=pl.when(
                    pl.col.has_historical_data
                )
                .then(
                    pl.col.category_historical_year_sales.lt(
                        LOW_CATEGORY_HISTORICAL_SALES
                    )
                )
                .otherwise(True)
            )
            .with_columns(
                new_category_problem=(
                    (pl.col.uses_po | pl.col.uses_ce | pl.col.uses_ne)
                    & pl.col.category_marked_new
                    & ~(pl.col.has_e_data | pl.col.has_po_data)
                ),
                new_print_problem=(
                    (
                        (pl.col.uses_po | pl.col.uses_ce)
                        & ~pl.col.category_marked_new
                        & pl.col.is_new_sku
                        & ~(pl.col.has_e_data | pl.col.has_po_data)
                    )
                    | (
                        (pl.col.uses_e | pl.col.uses_ce)
                        & pl.col.is_new_sku
                        & (
                            pl.col.low_current_period_sales
                            | pl.col.low_category_historical_sales
                            | ~pl.col.has_e_data
                        )
                    )
                ),
                po_problem=(
                    pl.col.uses_po
                    & ~pl.col.has_po_data
                    & ~pl.col.category_marked_new
                ),
                e_problem=(pl.col.uses_e & ~pl.col.has_e_data),
                ce_problem=(
                    pl.col.uses_ce & ~pl.col.has_e_data & ~pl.col.has_po_data
                ),
                ce_forced_to_use_po=(
                    pl.col.uses_ce & pl.col.has_po_data & ~pl.col.has_e_data
                ),
                ce_forced_to_use_e=(
                    pl.col.uses_ce & ~pl.col.has_po_data & pl.col.has_e_data
                ),
            )
        )

        find_dupes(input_data_info, ALL_SKU_AND_CHANNEL_IDS, raise_error=True)

        return input_data_info

    def predict_demand(
        self,
        channels: ScalarOrList[str] | ScalarOrList[Channel],
        start_date: DateLike,
        end_date: DateLike,
        aggregate_final_result: bool = True,
        force_po_prediction: bool = False,
    ) -> pl.DataFrame:
        """Predict demand across all SKUs and channels."""
        channels = [Channel.parse(ch) for ch in normalize_as_list(channels)]
        expected_demands_per_channel = {}

        input_data_info = self.get_input_data_info(
            force_po_prediction=force_po_prediction
        )

        if self.db.dfs[DataVariant.InStockRatio] is not None:
            mean_current_period_isr = (
                self.db.dfs[DataVariant.InStockRatio]
                .select(
                    ["sku", "a_sku"]
                    + Channel.members()
                    + ["date", "in_stock_ratio"]
                )
                .join(
                    input_data_info.select("sku", "a_sku", "current_period"),
                    on=["sku", "a_sku"],
                )
                .filter(
                    pl.col.date.ge(pl.col.current_period.list.first())
                    & pl.col.date.lt(pl.col.current_period.list.last())
                )
                .group_by(["sku", "a_sku"] + Channel.members())
                .agg(
                    pl.col.in_stock_ratio.mean().alias(
                        "mean_current_period_isr"
                    )
                )
            )
        else:
            mean_current_period_isr = None

        match_main_program_month_fractions = (
            self.analysis_defn.match_main_program_month_fractions
            if isinstance(self.analysis_defn, FbaRevDefn)
            else False
        )
        for channel in channels:
            expected_demands_per_group: list[CategoryGroup[pl.DataFrame]] = []
            for pdi in self.get_category_groups_for_channel(
                channel
            ).category_groups():
                expected_demand_from_history = pdi.expected_demand_from_history(
                    self.db.meta_info.all_sku,
                    start_date,
                    end_date,
                    match_main_program_month_fractions=match_main_program_month_fractions,
                )

                expected_demand_from_po = (
                    pdi.po_prediction.expected_demand_in_period(
                        start_date,
                        end_date,
                        match_main_program_month_fractions=match_main_program_month_fractions,
                    )
                ).select(
                    ["a_sku", "sku"]
                    + Channel.members()
                    + [
                        "date",
                        "monthly_expected_demand_from_po",
                        "expected_demand_from_po",
                    ]
                )

                assert (
                    expected_demand_from_history[
                        "expected_demand_from_history"
                    ].dtype
                    == pl.Int64()
                )
                assert (
                    expected_demand_from_po["expected_demand_from_po"].dtype
                    == pl.Int64()
                )

                expected_demand_df = (
                    join_and_coalesce(
                        expected_demand_from_history,
                        expected_demand_from_po,
                        NoOverride(),
                    )
                    .join(
                        input_data_info,
                        on=["a_sku", "sku"] + Channel.members(),
                        how="left",
                        validate="m:1",
                        join_nulls=True,
                    )
                    .with_columns(
                        expected_demand=pl.lit(None, dtype=pl.Int64())
                    )
                )

                data_existence_issue = expected_demand_df.select(
                    ["sku", "a_sku"]
                    + Channel.members()
                    + [
                        "expected_demand_from_po",
                        "expected_demand_from_history",
                        "has_po_data",
                        "has_e_data",
                    ]
                ).with_columns(
                    po_data_exists_issue=pl.col.has_po_data
                    & pl.col.expected_demand_from_po.is_null(),
                    e_data_exists_issue=pl.col.has_e_data
                    & pl.col.expected_demand_from_history.is_null(),
                )
                assert not data_existence_issue["po_data_exists_issue"].any()
                assert not data_existence_issue["e_data_exists_issue"].any()

                assert len(expected_demand_df) > 0

                performance_flag_df = (
                    expected_demand_df.filter(
                        pl.col.date.dt.month().eq(
                            expected_demand_df["date"]
                            .dt.month()
                            .sort()
                            .first()
                        )
                    )
                    .select(
                        ["a_sku", "sku"]
                        + Channel.members()
                        + [
                            "has_po_data",
                            "monthly_expected_demand_from_history",
                            "monthly_expected_demand_from_po",
                        ]
                    )
                    .with_columns(
                        enable_performance_flag=pl.lit(
                            self.analysis_defn.overperformer_settings.active,
                            dtype=pl.Boolean(),
                        )
                    )
                    .with_columns(
                        performance_flag=pl.when(
                            pl.col.enable_performance_flag
                        )
                        .then(
                            pl.when(
                                pl.col.monthly_expected_demand_from_history.is_not_null()
                                & pl.col.has_po_data
                                & (
                                    pl.col.monthly_expected_demand_from_history
                                    > (
                                        (1 + OUTPERFORM_FACTOR)
                                        * pl.col.monthly_expected_demand_from_po
                                    )
                                )
                            )
                            .then(PerformanceFlag.OVER.polars_lit())
                            .otherwise(
                                pl.when(
                                    pl.col.monthly_expected_demand_from_history.is_not_null()
                                    & pl.col.has_po_data
                                    & (
                                        pl.col.monthly_expected_demand_from_history
                                        < (
                                            (1 - OUTPERFORM_FACTOR)
                                            * pl.col.monthly_expected_demand_from_po
                                        )
                                    )
                                )
                                .then(PerformanceFlag.UNDER.polars_lit())
                                .otherwise(PerformanceFlag.NORMAL.polars_lit())
                            )
                        )
                        .otherwise(PerformanceFlag.DISABLED.polars_lit())
                    )
                    .select(
                        ["a_sku", "sku"]
                        + Channel.members()
                        + ["performance_flag"]
                    )
                )

                overperformers = performance_flag_df.filter(
                    pl.col.performance_flag.eq(PerformanceFlag.OVER.name)
                )

                overperformer_demand = (
                    pdi.expected_demand_from_history(
                        self.db.meta_info.all_sku,
                        start_date,
                        self.analysis_defn.overperformer_settings.prediction_offset.apply_to(
                            start_date
                        ),
                        filter_a_sku=overperformers["a_sku"].unique(),
                        aggregate=True,
                        match_main_program_month_fractions=(
                            match_main_program_month_fractions
                        ),
                    )
                    .with_columns(
                        uses_overperformer_estimate=pl.lit(
                            True, dtype=pl.Boolean()
                        )
                    )
                    .rename(
                        {"expected_demand_from_history": "expected_demand"}
                    )
                )

                if mean_current_period_isr is not None:
                    expected_demand_df = expected_demand_df.join(
                        mean_current_period_isr,
                        on=["sku", "a_sku"] + Channel.members(),
                        how="left",
                    )
                else:
                    expected_demand_df = expected_demand_df.with_columns(
                        mean_current_period_isr=pl.lit(
                            None, dtype=polars_float(64)
                        )
                    )

                if self.analysis_defn.enable_low_current_period_isr_logic:
                    expected_demand_df = expected_demand_df.with_columns(
                        has_low_isr=pl.when(
                            pl.col.mean_current_period_isr.is_not_null()
                            & pl.col.mean_current_period_isr.lt(0.5)
                        )
                        .then(pl.lit(True))
                        .otherwise(pl.lit(False))
                    )
                else:
                    expected_demand_df = expected_demand_df.with_columns(
                        has_low_isr=pl.lit(False),
                    )

                expected_demand_df = (
                    expected_demand_df.drop(
                        [
                            "monthly_expected_demand_from_history",
                            "monthly_expected_demand_from_po",
                        ]
                    )
                    .join(
                        performance_flag_df,
                        on=["a_sku", "sku"] + Channel.members(),
                        how="left",
                    )
                    .with_columns(
                        e_overrides_po=(
                            pl.col.uses_po
                            & ~pl.col.uses_ne
                            & pl.col.po_problem
                        ),
                        po_overrides_e=(
                            pl.col.uses_e
                            & pl.col.expected_demand_from_po.is_not_null()
                            & ~pl.col.has_e_data
                        ),
                    )
                    .with_columns(
                        ce_uses_e=(
                            pl.col.uses_ce
                            & (
                                pl.col.ce_forced_to_use_e
                                | (
                                    pl.col.has_e_data
                                    & pl.col.has_po_data
                                    & (
                                        (
                                            (
                                                pl.col.expected_demand_from_history
                                                < 0.5
                                                * pl.col.expected_demand_from_po
                                            )
                                            & pl.col.is_new_sku
                                        )
                                        | (
                                            pl.col.expected_demand_from_history
                                            > pl.col.expected_demand_from_po
                                        )
                                    )
                                )
                            )
                        ),
                    )
                    .with_columns(
                        ce_uses_po=(
                            pl.col.uses_ce
                            & (~pl.col.ce_uses_e | pl.col.ce_forced_to_use_po)
                        ),
                    )
                    .with_columns(
                        expected_demand=pl.when(
                            (pl.col.uses_e & ~pl.col.po_overrides_e)
                            | pl.col.ce_uses_e
                            | pl.col.e_overrides_po
                        )
                        .then(pl.col.expected_demand_from_history)
                        .otherwise(pl.col.expected_demand)
                    )
                    .with_columns(
                        expected_demand=pl.when(
                            (pl.col.uses_po & ~pl.col.e_overrides_po)
                            | pl.col.ce_uses_po
                            | pl.col.po_overrides_e
                            # | pl.lit(force_po_prediction)
                        )
                        .then(pl.col.expected_demand_from_po)
                        .otherwise(pl.col.expected_demand)
                    )
                    .with_columns(
                        expected_demand=pl.when(
                            (
                                # ~pl.lit(force_po_prediction) &
                                pl.col.uses_ne
                                | (pl.col.uses_e & pl.col.has_low_isr)
                            )
                        )
                        .then(
                            pl.max_horizontal(
                                "expected_demand_from_history",
                                "expected_demand_from_po",
                            )
                        )
                        .otherwise(pl.col.expected_demand)
                    )
                    .with_columns(
                        demand_based_on_e=pl.col("expected_demand").eq(
                            pl.col.expected_demand_from_history
                        ),
                        demand_based_on_po=pl.col("expected_demand").eq(
                            pl.col.expected_demand_from_po
                        ),
                    )
                )

                assert (
                    expected_demand_df["expected_demand"].dtype == pl.Int64()
                )

                if aggregate_final_result:
                    expected_demand_df = (
                        expected_demand_df.group_by(
                            ["sku", "a_sku"] + Channel.members()
                        )
                        .agg(
                            pl.col.date,
                            pl.col.date.len().alias("prediction_parts"),
                            pl.col.e_overrides_po.sum(),
                            pl.col.po_overrides_e.sum(),
                            pl.col.ce_uses_e.sum(),
                            pl.col.ce_uses_po.sum(),
                            pl.col.has_low_isr.sum(),
                            pl.col.demand_based_on_e.sum(),
                            pl.col.demand_based_on_po.sum(),
                            pl.col.mean_current_period_isr,
                            pl.col.expected_demand_from_history,
                            pl.col.expected_demand_from_po,
                            pl.col.expected_demand.sum(),
                            pl.col.performance_flag.first(),
                            pl.col.prediction_type,
                        )
                        .with_columns(
                            uses_overperformer_estimate=pl.lit(
                                False, dtype=pl.Boolean()
                            )
                        )
                    )

                    expected_demand_df = join_and_coalesce(
                        expected_demand_df,
                        overperformer_demand,
                        OverrideLeft(["a_sku", "sku"] + Channel.members()),
                    )
                else:
                    assert not self.analysis_defn.overperformer_settings.active

                expected_demands_per_group.append(
                    CategoryGroup(pdi.strategy, expected_demand_df)
                )

            expected_demands_per_channel[channel] = CategoryGroups(
                expected_demands_per_group
            )

        collated_result = collate_groups_per_channel(
            expected_demands_per_channel,
            # dupe_check_index=["a_sku", "sku"] + Channel.members(),
        )

        if aggregate_final_result:
            find_dupes(
                collated_result,
                ["sku", "a_sku"] + Channel.members(),
                raise_error=True,
            )
        else:
            find_dupes(
                collated_result,
                ["sku", "a_sku"] + Channel.members() + ["date"],
                raise_error=True,
            )
        if not force_po_prediction:
            collated_result = collated_result.drop("prediction_type")

        return collated_result
