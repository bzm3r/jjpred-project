"""Aggregators for consolidating sales information within a dataframe according
to given criteria.
"""

from __future__ import annotations

from collections.abc import Callable
import polars as pl


from jjpred.channel import Channel, DistributionMode
from jjpred.countryflags import CountryFlags
from jjpred.utils.typ import (
    ScalarOrList,
    normalize_as_list,
    normalize_optional,
)


class Aggregator:
    """Represents a function which takes a sales information dataframe, and
    returns an aggregated dataframe.
    """

    description: str
    filter_expr: pl.Expr

    def __init__(self, filter_expr: pl.Expr, description: str = "") -> None:
        self.filter_expr = filter_expr
        self.description = description

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError()

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggregates the sales in the given dataframe, by category, across the
        channels filtered by this Aggregator's ``filter_expr``, dataframe.

        Args:
            df (pl.DataFrame): dataframe containing sales data.

        Returns:
            pl.DataFrame
        """
        df = df.filter(self.filter_expr)
        return df.group_by(
            "category",
            "date",
        ).agg(pl.col("sales").sum())

    def __repr__(self) -> str:
        return self.description.__repr__()


type AggregatorLike = Callable[[pl.DataFrame], pl.DataFrame] | Aggregator


class UsingChannels(Aggregator):
    """Aggregates sales by category across ``focus_channels`` channels
    in the given sales dataframe. If ``focus_channels`` is ``None``, then it
    does not filter at all."""

    focus_channels: list[Channel] | None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.focus_channels == other.focus_channels
        return False

    def __init__(
        self, focus_channels: ScalarOrList[Channel | str] | None = None
    ):
        """Aggregates sales by category across all retail channels
        in the given sales dataframe.

        :param focus_channels: _description_, defaults to None
        :type focus_channels: ScalarOrList[Channel  |  str] | None, optional
        """
        if focus_channels is not None:
            self.focus_channels = [
                Channel.parse(ch) for ch in normalize_as_list(focus_channels)
            ]
            filter_expr = pl.struct(Channel.members()).is_in(
                pl.Series(
                    [x.as_dict() for x in self.focus_channels],
                    dtype=Channel.polars_type_struct(),
                ).implode()
            )
        else:
            self.focus_channels = focus_channels
            filter_expr = pl.lit(True).eq(pl.lit(True))

        focus_channels_reprs = [
            str(ch) for ch in normalize_optional(self.focus_channels, [])
        ]
        super().__init__(
            filter_expr,
            description="agg: " + f"filter_channels: {focus_channels_reprs}",
        )


class UsingCanUSRetail(Aggregator):
    """Aggregates sales by category across all CA/US retail channels
    in the given sales dataframe."""

    def __init__(self):
        super().__init__(
            pl.col("country_flag")
            .and_(int(CountryFlags.CA | CountryFlags.US))
            .gt(0)
            & pl.col("mode").eq(DistributionMode.RETAIL.name),
            description="agg: retail, all can/us channels",
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)
