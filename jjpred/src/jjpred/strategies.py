"""Strategies determine what historical periods, current periods, historical
data aggregation criteria, default category prediction types, and reference categories."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from textwrap import TextWrapper
from typing import Any, Self
import polars as pl
import polars.selectors as cs

from jjpred.aggregator import Aggregator
from jjpred.analysisdefn import AnalysisDefn
from jjpred.channel import Channel
from jjpred.inputstrategy import (
    InputStrategy,
    ContiguousTimePeriod,
    UndeterminedTimePeriod,
)
from jjpred.sku import Category

from jjpred.strategy import CategoryType
from jjpred.utils.datetime import Date, DateLike
from jjpred.utils.fileio import gen_meta_info_path
from jjpred.utils.groupeddata import (
    CategoryGroupProtocol,
    CategoryGroups,
    ChannelCategoryData,
)
from jjpred.utils.multidict import MultiDict
from jjpred.utils.polars import find_dupes, vstack_to_unified, struct_filter
from jjpred.utils.typ import (
    as_polars_type,
)


@dataclass
class HistoricalDfs:
    historical_df: pl.DataFrame
    """The historical year whose sales data we use to generate initial monthly
    ratios."""
    working_df: pl.DataFrame
    """The working year whose sales data we use to adjust monthly ratios from
    the historical year."""
    months_missing_from_working_year: list[int]
    """Incomplete months from the working year."""


@dataclass
class StrategyGroup(CategoryGroupProtocol):
    parent: StrategyGroups
    channel: Channel  # strategy groups are specific to a channel we are making a prediction for
    primary_cats: list[Category] = field(compare=False, hash=False)
    current_periods: MultiDict[
        Category, ContiguousTimePeriod | UndeterminedTimePeriod
    ] = field(compare=False, hash=False)
    aggregator: Aggregator = field(compare=True, hash=True)
    primary_to_referred: dict[Category, list[Category]] = field(
        default_factory=dict, compare=False, hash=False, init=False
    )
    __all_categories__: list[Category] = field(
        default_factory=list, compare=False, hash=False, init=False
    )
    __rebuild_category_types__: bool = field(
        default=True, compare=False, hash=False, init=False
    )
    __category_types__: dict[CategoryType, list[Category]] | None = field(
        default=None, compare=False, hash=False, init=False
    )

    def __post_init__(self):
        assert len(self.all_categories) == 0
        self.__all_categories__ += self.primary_cats

    def __or__(self, other: Any) -> Self:
        if isinstance(other, self.__class__):
            if self == other:
                self.primary_cats += other.primary_cats
                return self
            else:
                raise ValueError(
                    f"Cannot union, strategies unequal:\n{self=}\n{other=}"
                )
        elif other is None:
            return self
        else:
            raise ValueError(f"Cannot union: {self} | {other}")

    def __ror__(self, other: Any) -> Self:
        return self.__or__(other)

    def get_reference_category(
        self, referred_category: Category
    ) -> Category | None:
        for primary, referred_cats in self.primary_to_referred.items():
            if referred_category in referred_cats:
                return primary

    def insert_referred_category(
        self, primary: Category, referred: list[Category]
    ):
        referred = sorted(referred)
        if primary not in self.primary_to_referred.keys():
            self.primary_to_referred[primary] = referred
        else:
            for r in referred:
                if r not in self.primary_to_referred[primary]:
                    bisect.insort(self.primary_to_referred[primary], r)

        self.__all_categories__ += referred

    def combine_group(self, other: StrategyGroup):
        self.primary_cats += other.primary_cats
        self.__all_categories__ += other.__all_categories__
        self.__rebuild_category_types__ = True
        self.current_periods = self.current_periods | other.current_periods
        assert self.aggregator == other.aggregator
        # for k, v in other.primary_to_referred.items():
        #     self.insert_referred_category(k, v)
        assert (
            self.primary_to_referred == {} and other.primary_to_referred == {}
        )

    @property
    def category_types(
        self,
    ) -> dict[CategoryType, list[Category]]:
        if self.__category_types__ is None or self.__rebuild_category_types__:
            self.__category_types__ = (
                self.__partition_category_type_by_dependency__()
            )

        return self.__category_types__

    def get_category_type(self, category: Category) -> CategoryType | None:
        for ref_cat_type, cats in self.category_types.items():
            if category in cats:
                return ref_cat_type

    def __partition_category_type_by_dependency__(
        self,
    ) -> dict[CategoryType, list[Category]]:
        primaries = self.primary_cats

        dependents = []
        for primary in primaries:
            dependents += self.primary_to_referred.get(primary, [])
        dependents = sorted(dependents)
        all_cats = sorted(primaries + dependents)

        return {
            CategoryType.ALL: all_cats,
            CategoryType.PRIMARY: primaries,
            CategoryType.DEPENDENT: dependents,
        }

    @property
    def all_categories(self) -> list[Category]:
        if self.__all_categories__ is not None:
            return self.__all_categories__
        else:
            self.__all_categories__ = []
            for primary, ref_cats in self.primary_to_referred.items():
                self.__all_categories__ += [primary] + ref_cats

        return self.__all_categories__

    def map_primary_to_referred(
        self, df: pl.DataFrame
    ) -> dict[Category, pl.Series]:
        category_enum = as_polars_type(df["category"].dtype, pl.Enum)
        this_primary_categories = pl.Series(
            sorted(
                set(self.primary_to_referred.keys()).intersection(
                    df["category"].unique()
                )
            ),
            dtype=category_enum,
        )
        map_primary_to_referred = {
            k: pl.Series("referred_cats", list(vs), dtype=category_enum)
            for k, vs in self.primary_to_referred.items()
            if k in this_primary_categories
        }
        return map_primary_to_referred

    def append_referred_channels(self, df: pl.DataFrame) -> pl.DataFrame:
        if len(self.primary_to_referred) > 0:
            map_primary_to_referred = self.map_primary_to_referred(df)
            extra = (
                df.filter(
                    pl.col("category").is_in(map_primary_to_referred.keys())
                )
                .with_columns(
                    pl.col("category").replace_strict(
                        map_primary_to_referred,
                        return_dtype=pl.List(
                            as_polars_type(df["category"].dtype, pl.Enum)
                        ),
                    )
                )
                .explode("category")
            )
            return vstack_to_unified(df, extra)

        return df

    def construct_history(
        self,
        history_df: pl.DataFrame,
        dispatch_date: Date,
    ) -> HistoricalDfs:
        history_time_period = ContiguousTimePeriod.make_history_period(
            dispatch_date
        )
        historical = self.aggregator(
            history_df.filter(
                pl.col("category").is_in(self.primary_cats),
                pl.col("date").is_in(
                    history_time_period.historical_year.tpoints
                ),
            ),
        ).rename({"sales": "category_historical_monthly_sales"})
        working = self.aggregator(
            history_df.filter(
                pl.col("category").is_in(self.primary_cats),
                pl.col("date").is_in(history_time_period.working_year.tpoints),
            ),
        ).rename({"sales": "category_working_monthly_sales"})

        historical = self.append_referred_channels(historical)
        working = self.append_referred_channels(working)

        return HistoricalDfs(
            historical,
            working,
            history_time_period.months_missing_from_working_year,
        )

    def construct_po_prediction(
        self,
        po_data: pl.DataFrame,
    ) -> pl.DataFrame:
        monthly_expected_demand_from_po = (
            # get PO data for this particular channel
            struct_filter(
                # get PO data only for the SKUs in the categories we require
                po_data.filter(
                    pl.col("category").is_in(
                        self.category_types[CategoryType.PRIMARY]
                        + self.category_types[CategoryType.DEPENDENT]
                    )
                )  # combine any sales information for the same item
                .group_by(cs.exclude("sales"))
                .agg(pl.col("sales").sum()),
                self.channel,
            )
            .rename({"sales": "monthly_expected_demand_from_po"})
            .with_columns(pl.col("monthly_expected_demand_from_po"))
        )

        return monthly_expected_demand_from_po

    def construct_current(
        self,
        default_current_period_end_date: DateLike,
        sales_history_df: pl.DataFrame,
    ) -> MultiDict[Category, pl.DataFrame]:
        history_df = struct_filter(sales_history_df, self.channel)
        # all_cats = self.all_categories

        result = MultiDict({})
        for categories, current_period in self.current_periods.data.items():
            current_period = current_period.with_end_date(
                default_current_period_end_date
            )
            self.current_periods.data[categories] = current_period
            result.data[categories] = history_df.filter(
                pl.col("category").is_in(categories),
                pl.col("date").is_in(current_period.tpoints),
            ).rename({"sales": "current_monthly_sales"})

        return result

    def is_similar(self, other) -> bool:
        return (
            self.channel == other.channel
            and self.aggregator == other.aggregator
        )

    def __repr__(self) -> str:
        result = []
        result.append(self.__class__.__qualname__ + "(")
        for k in self.__dataclass_fields__.keys():
            if k != "parent":
                field_repr = TextWrapper(
                    initial_indent="\t",
                    subsequent_indent="\t\t",
                    break_on_hyphens=False,
                ).fill(getattr(self, k).__repr__())
                result.append(f"{k}: {field_repr}")
        result.append(self.__class__.__qualname__ + ")")
        return "\n".join(result)


@dataclass
class CategoryGroup[T](CategoryGroupProtocol):
    strategy: StrategyGroup
    data: T

    @property
    def all_categories(self) -> list[Category]:
        return self.strategy.all_categories


def collate_groups(
    category_groups: CategoryGroups[CategoryGroup[pl.DataFrame]],
    dupe_check_index: list[str] | None = None,
) -> pl.DataFrame:
    stacked_df = pl.DataFrame()

    groups = category_groups.category_groups()
    if len(groups) > 0:
        stacked_df = groups[0].data

        for group in groups[1:]:
            if dupe_check_index is not None and len(dupe_check_index) > 0:
                find_dupes(group.data, dupe_check_index, raise_error=True)
            stacked_df = stacked_df.vstack(group.data)

            if dupe_check_index is not None and len(dupe_check_index) > 0:
                find_dupes(stacked_df, dupe_check_index, raise_error=True)

    return stacked_df


def collate_groups_per_channel(
    results_per_channel: dict[
        Channel, CategoryGroups[CategoryGroup[pl.DataFrame]]
    ],
    attach_channel: bool = False,
    dupe_check_index: list[str] | None = None,
) -> pl.DataFrame:
    result = pl.DataFrame()

    for ch, group_dfs in results_per_channel.items():
        collated = collate_groups(group_dfs, dupe_check_index=dupe_check_index)

        if dupe_check_index is not None and len(dupe_check_index) > 0:
            find_dupes(
                collated,
                dupe_check_index,
                raise_error=True,
            )

        if attach_channel:
            collated = collated.with_columns(**ch.to_columns())
        result = vstack_to_unified(result, collated)

    return result


class StrategyGroups(CategoryGroups[StrategyGroup]):
    channel: Channel  # StrategyGroups are specific to a channel
    all_categories: pl.Series

    def __init__(
        self,
        channel: Channel,
        all_categories: pl.Series,
        input_strategy: InputStrategy,
    ):
        self.channel = channel

        map_primary_to_referred: dict[Category, list[Category]] = {}
        referred_cats = []
        for (
            referred,
            primary,
        ) in input_strategy.referred_to_primary_map.items():
            if primary in map_primary_to_referred.keys():
                if referred not in map_primary_to_referred[primary]:
                    map_primary_to_referred[primary].append(referred)
                    referred_cats.append(referred)
            else:
                map_primary_to_referred[primary] = [referred]
                referred_cats.append(referred)

        self.data = []
        for category in [x for x in all_categories if x not in referred_cats]:
            self.file_primary_category(
                category,
                MultiDict.from_dict(
                    {
                        c: input_strategy.current_periods[c]
                        for c in map_primary_to_referred.get(category, [])
                        + [category]
                    }
                ),
                input_strategy.aggregators[category],
            )

        for primary, referred in map_primary_to_referred.items():
            for group in self.data:
                if primary in group.primary_cats:
                    group.insert_referred_category(primary, referred)
                    break

    def file_primary_category(
        self,
        category: Category,
        current_periods: MultiDict[
            Category, ContiguousTimePeriod | UndeterminedTimePeriod
        ],
        aggregator: Aggregator,
    ):
        inserted = False
        new_group = StrategyGroup(
            self,
            self.channel,
            [category],
            current_periods,
            aggregator,
        )

        inserted = False
        for x in self.data:
            if x.is_similar(new_group):
                x.combine_group(new_group)
                inserted = True
                break
        if not inserted:
            self.data.append(new_group)


class ChannelStrategyGroups(
    ChannelCategoryData[StrategyGroups, StrategyGroup]
):
    def __init__(
        self,
        analysis_defn: AnalysisDefn,
        input_strategies: list[InputStrategy],
    ):
        active_sku_info = pl.read_parquet(
            gen_meta_info_path(analysis_defn, "active_sku"),
            memory_map=False,
        )
        all_categories = pl.Enum(
            list(active_sku_info["category"].unique().sort())
        )
        # all_categories = as_polars_type(
        #     pl.Enum(active_sku_info["category"].unique().sort()), pl.Enum
        # )

        for input_strategy in input_strategies:
            for channel in input_strategy.channels:
                strategies = StrategyGroups(
                    channel, all_categories.categories, input_strategy
                )
                self.data[channel] = strategies
