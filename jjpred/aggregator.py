"""Aggregators for consolidating sales information within a dataframe according
to given criteria.
"""

from __future__ import annotations

from collections.abc import Callable
import polars as pl


from jjpred.channel import Channel, DistributionMode
from jjpred.countryflags import CountryFlags
from jjpred.utils.polars import FilterStructs, struct_filter
from jjpred.utils.typ import (
    ScalarOrList,
    as_list,
    normalize_optional,
)


class Aggregator:
    """Represents a function which takes a sales information dataframe, and
    returns an aggregated dataframe.
    """

    description: str

    def __init__(self, description: str = "") -> None:
        self.description = description

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError()

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return self.description.__repr__()


type AggregatorLike = Callable[[pl.DataFrame], pl.DataFrame] | Aggregator


def agg_retail(history: pl.DataFrame) -> pl.DataFrame:
    """Aggregates historical sales by category across all retail channels in the
    given historical sales dataframe.
    """
    return (
        history.filter(pl.col("mode").eq(DistributionMode.RETAIL.name))
        .group_by(
            "category",
            "date",
        )
        .agg(pl.col("sales").sum())
    )


class UsingRetail(Aggregator):
    """Aggregates sales by category across ``focus_channels`` retail channels
    in the given sales dataframe. If ``focus_channels`` is ``None``, then it
    filters for retail channels alone."""

    focus_channels: list[Channel] | None
    channel_filter: FilterStructs | None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.focus_channels == other.focus_channels
        return False

    def __init__(
        self, focus_channels: ScalarOrList[Channel | str] | None = None
    ):
        """Aggregates sales by category across all retail channels
        in the given sales dataframe.

        Args:
            focus_channels (ScalarOrList[Channel | str] | None): optional
            list of channels to filter for in particular.
        Returns:
            UsingRetail
        """
        if focus_channels is not None:
            self.focus_channels = [
                Channel.parse(ch) for ch in as_list(focus_channels)
            ]
            self.channel_filter = FilterStructs(set(self.focus_channels))
        else:
            self.focus_channels = focus_channels
            self.channel_filter = None

        focus_channels_reprs = [
            str(ch) for ch in normalize_optional(self.focus_channels, [])
        ]
        super().__init__(
            "agg: retail, " + f"filter_channels: {focus_channels_reprs}"
        )

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggregates sales by category across all retail channels
        in the given sales dataframe.

        Args:
            df (pl.DataFrame): dataframe containing sales data.

        Returns:
            pl.DataFrame
        """
        df = struct_filter(df, self.channel_filter)
        return agg_retail(df)


class UsingCanUSRetail(Aggregator):
    """Aggregates sales by category across all CA/US retail channels
    in the given sales dataframe."""

    def __init__(self):
        super().__init__("agg: retail, all can/us channels")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggregates sales by category across all CA/US retail channels
        in the given sales dataframe.

        Args:
            df (pl.DataFrame): dataframe containing sales data.

        Returns:
            pl.DataFrame
        """
        df = df.filter(
            pl.col("country_flag")
            .and_(int(CountryFlags.CA | CountryFlags.US))
            .gt(0),
            pl.col("mode").eq(DistributionMode.RETAIL.name),
        )
        return agg_retail(df)


class UsingAllRetail(UsingRetail):
    """Aggregates sales by category across all retail channels
    in the given sales dataframe."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggregates sales by category across all retail channels
        in the given sales dataframe.

        Args:
            df (pl.DataFrame): dataframe containing sales data.

        Returns:
            pl.DataFrame
        """
        return agg_retail(df)


class UsingAllChannels(Aggregator):
    """Aggregates sales by category across all channels
    in the given sales dataframe. (I.e. it does not filtering at all.)"""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.group_by(
            "category",
            "date",
        ).agg(pl.col("sales").sum())
