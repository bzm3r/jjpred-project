"""Channels are the various locations where J&J products are sold."""

from __future__ import annotations

from collections import UserList
from dataclasses import dataclass, field
from enum import auto
import re
from typing import (
    Any,
    NamedTuple,
    Protocol,
    Self,
    cast,
    runtime_checkable,
)
import polars as pl
from jjpred.countryflags import CountryFlags
from jjpred.parse.patternmatch import (
    CompiledMatchSkip,
    ReMatchCondition,
    ReMatchResult,
    ReMatcher,
    StringPattern,
)
from jjpred.structlike import FieldMeta, MemberType, StructLike
from jjpred.utils.polars import EnumLike
from jjpred.utils.typ import (
    ScalarOrList,
    do_nothing,
    normalize_optional,
)


class DistributionMode(EnumLike):
    """:py:class:`EnumLike` representing different modes a channel might
    operate under."""

    RETAIL = auto()
    """Retail distribution."""
    WHOLESALE = auto()
    """Wholesale distribution."""
    WAREHOUSE = auto()
    """Warehouse."""
    NO_MODE = auto()


# TODO: figure out why using anything but a pl.Int64() fails.
assert len([x for x in CountryFlags]) <= 16
PolarsCountryFlagType = pl.Int64()  # pl.UInt16()


@runtime_checkable
class CountryInfo(Protocol):
    """Abstract class representing different ways in which a channel might
    have country information."""

    def contains_any(self, country: CountryFlags) -> bool:
        """Does this object contain the given country?

        Args:
            country (Country): country to check for.

        Returns:
            bool: whether the country is in this object.
        """
        raise NotImplementedError()


@dataclass
class PossibleCountries(UserList[CountryFlags], CountryInfo):
    """Some platforms operate in multiple countries, so this object allows us to
    represent the possible countries such platforms might operate in."""

    data: list[CountryFlags]

    def contains_any(self, country: CountryFlags) -> bool:
        for x in self:
            if country in x:
                return True
        return False


@dataclass
class FixedCountries(CountryInfo):
    """Some platforms only operate in a known/fixed set of countries, and this
    object lets us represent these countries."""

    fixed: CountryFlags

    def contains_any(self, country: CountryFlags) -> bool:
        return country in self.fixed


class ChannelMatchResult(ReMatchResult):
    """Channel match results pay special to attention how country information is
    recorded."""

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, str) and value != "":
            if key.startswith("country_"):
                if any(x.startswith("country_flag") for x in self.keys()):
                    raise KeyError(
                        f"Key with 'country*' form already exists: {self.data}"
                        f"; attempting to insert ({key}, {value})"
                    )
                else:
                    self.data["country_flag"] = key.split("_")[-1]
            else:
                self.data[key] = value


class ChannelMatcher(ReMatcher):
    """Matcher for parsing strings which represent a channel."""

    platform_fragments: list[str]
    """Strings that should be matched as this channel's platform."""
    country_info: PossibleCountries | FixedCountries | None
    """Country information associated with this channel."""

    def __init__(
        self,
        platform_fragments: list[str],
        countries: PossibleCountries | FixedCountries | None,
        skip_patterns: ScalarOrList[str] | None = None,
    ):
        skip_patterns = normalize_optional(skip_patterns, [])
        self.platform_fragments = platform_fragments
        platform_pattern = (
            StringPattern()
            .any_of(
                cast(ScalarOrList[str | StringPattern], platform_fragments)
            )
            .named("platform")
            .fragmentlike(
                start_patterns=[
                    r"^",
                    r"\b",
                    r"_",
                ],
                end_patterns=[r"\.", r" - ", r" "],
            )
        )

        self.country_info = countries
        all_country_patterns = None
        if countries is not None:
            if isinstance(countries, PossibleCountries):
                country_patterns = [c.string_pattern() for c in countries]
                extra_token_boundaries = [r"_", r".", r" - ", r"-"]
                all_country_patterns = (
                    StringPattern()
                    .any_of(*country_patterns)
                    .fragmentlike(
                        start_patterns=cast(
                            list[str | StringPattern], extra_token_boundaries
                        ),
                        end_patterns=cast(
                            list[str | StringPattern], extra_token_boundaries
                        ),
                    )
                    .no_capture()
                    .optional()
                )

        if all_country_patterns is not None:
            final_pattern = platform_pattern.concatenate(
                StringPattern(all_country_patterns),
                joiner=StringPattern()
                .any_of(r"\b", r"\.", r"_", r" - ", r"-")
                .no_capture(),
            )
        else:
            final_pattern = platform_pattern.concatenate(StringPattern(r".*"))

        match_pattern = final_pattern.named("channel").compile(
            ChannelMatchResult, flags=re.IGNORECASE
        )

        skip_pattern = (
            StringPattern()
            .any_of(r"^s_", cast(list[str | StringPattern], skip_patterns))
            .named("skip")
        )
        if not skip_pattern.is_empty():
            compiled_skip = skip_pattern.compile(
                ChannelMatchResult, flags=re.IGNORECASE
            )
        else:
            compiled_skip = None

        self.match_skip = CompiledMatchSkip(match_pattern, compiled_skip)
        super().__init__("channel", self.match_skip, ReMatchCondition.DeepAll)


class PlatformAttrs(NamedTuple):
    """Various meta information associated with a platform."""

    channel_matcher: ChannelMatcher
    distribution_mode: DistributionMode


class Platform(PlatformAttrs, EnumLike):
    """A platform represents a group of related channels. For example: "Amazon"
    or "HongMall"."""

    Amazon = (
        ChannelMatcher(
            ["amazon", "amz"],
            PossibleCountries([case for case in CountryFlags]),
        ),
        DistributionMode.RETAIL,
    )
    Faire = (
        ChannelMatcher(["faire"], FixedCountries(CountryFlags.US)),
        DistributionMode.WHOLESALE,
    )
    HongMall = (
        ChannelMatcher(
            ["hongmall"],
            PossibleCountries([CountryFlags.CA, CountryFlags.US]),
        ),
        DistributionMode.RETAIL,
    )
    JanAndJul = (
        ChannelMatcher(
            ["janandjul"],
            FixedCountries(CountryFlags.CA | CountryFlags.US),
            skip_patterns=[r"^inv_jj"],
        ),
        DistributionMode.RETAIL,
    )
    PopUp = (
        ChannelMatcher(
            [
                "pop-up shop",
                "popup",
                "vancouver showroom",
            ],
            FixedCountries(CountryFlags.CA),
        ),
        DistributionMode.RETAIL,
    )
    Bay = (
        ChannelMatcher(["thebay", "thebay"], FixedCountries(CountryFlags.CA)),
        DistributionMode.RETAIL,
    )
    Walmart = (
        ChannelMatcher(
            ["walmart"],
            PossibleCountries([CountryFlags.CA, CountryFlags.US]),
        ),
        DistributionMode.RETAIL,
    )
    Warehouse = (
        ChannelMatcher(
            ["inv_wh", "warehouse"],
            PossibleCountries([CountryFlags.CA, CountryFlags.CN]),
        ),
        DistributionMode.WAREHOUSE,
    )
    Wholesale = (
        ChannelMatcher(
            ["wholesale"],
            None,
        ),
        DistributionMode.WHOLESALE,
    )
    XiaoHongShu = (
        ChannelMatcher(
            ["xiaohongshu"],
            PossibleCountries(
                [CountryFlags.CA, CountryFlags.US, CountryFlags.CN]
            ),
        ),
        DistributionMode.RETAIL,
    )
    AllChannels = (
        ChannelMatcher(
            ["all channels"],
            FixedCountries(CountryFlags(CountryFlags.max_int())),
        ),
        DistributionMode.NO_MODE,
    )

    @classmethod
    def matcher(
        cls,
    ) -> ReMatcher:
        """Creates matcher to parse a string that might represent any channel."""
        return ReMatcher(
            "all_channels",
            [x.channel_matcher.match_skip for x in cls],
            ReMatchCondition.DeepAny,
        )

    @classmethod
    def amazon_matcher(
        cls,
    ) -> ReMatcher:
        """Creates a matcher to parse a string that might represent an Amazon channel."""
        return ReMatcher(
            "amazon_matcher",
            [
                x.channel_matcher.match_skip
                for x in cls
                if "amazon" in x.name.lower()
            ],
            ReMatchCondition.DeepAny,
        )

    @classmethod
    def from_str(cls, string: str) -> Self:
        low_string = string.lower()
        for x in cls:
            if x.name.lower() == low_string or x.channel_matcher.apply(string):
                return x

        raise ValueError(f"Cannot parse {string} as {cls.__qualname__}.")

    def __str__(self) -> str:
        return self.name


def country_parser(
    x: CountryFlags | FixedCountries | PossibleCountries | None,
) -> int | None:
    """Parse an object that represents a country into an integer representation
    (recall that countries are ``IntFlag``)."""
    if x:
        if isinstance(x, FixedCountries):
            return int(x.fixed)
        elif isinstance(x, CountryFlags):
            return int(x)
    else:
        return None

    raise ValueError(f"Cannot parse {x} as country.")


class Channel(
    StructLike,
    matcher=Platform.matcher(),
    joiner=" ",
):
    """Represents a specific channel where J&J goods are sold."""

    channel: str = field(
        default_factory=str,
        kw_only=True,
        hash=False,
        compare=False,
        metadata=FieldMeta(MemberType.META, do_nothing, pl.String()),
    )
    """The string which was parsed as this channel."""

    platform: str = field(
        kw_only=True,
        hash=False,
        compare=True,
        metadata=FieldMeta(
            MemberType.PRIMARY,
            do_nothing,
            Platform.polars_type(),
            intermediate_polars_dtype=pl.String(),
        ),
    )
    """The platform this channel is associated with."""

    country_flag: CountryFlags = field(
        default=CountryFlags(0),
        kw_only=True,
        hash=False,
        compare=True,
        metadata=FieldMeta(
            MemberType.PRIMARY, country_parser, PolarsCountryFlagType
        ),
    )
    """The country(ies) this channel is associated with."""

    mode: DistributionMode = field(
        default=DistributionMode.NO_MODE,
        kw_only=True,
        hash=False,
        compare=True,
        metadata=FieldMeta(
            MemberType.PRIMARY,
            str,
            DistributionMode.polars_type(),
            intermediate_polars_dtype=pl.String(),
        ),
    )
    """The distribution mode this channel operates in."""

    @classmethod
    def from_dict(cls, x: dict[str, str]) -> Channel:
        platform = Platform.try_from_str(x["platform"])
        if platform:
            country = CountryFlags.try_parse(x.get("country_flag"))
            if (country is None or country == CountryFlags(0)) and isinstance(
                platform.channel_matcher.country_info, FixedCountries
            ):
                country = platform.channel_matcher.country_info.fixed
            mode = platform.distribution_mode
            channel_dict: dict[str, Any] = {
                "channel": x.get("channel"),
                "platform": platform.name,
                "country_flag": country,
                "mode": mode,
            }
            return cls(**(cls.field_defaults | channel_dict))
        else:
            raise KeyError(f"No platform in: {x}")

    def pretty_string_repr(self) -> str:
        country_flag = (
            self.country_flag.try_to_string()
            if self.country_flag is not None
            else None
        )
        if country_flag is None:
            country_flag = "??"

        mode = self.mode.try_to_string()
        if mode is None:
            mode = "MODE=??"

        return f"{self.platform} {country_flag } {mode}"

    @classmethod
    def map_polars_struct_to_string(cls, polars_struct: Any) -> str:
        assert all([x in polars_struct.keys() for x in Channel.members()])
        return Channel.from_dict(polars_struct).pretty_string_repr()
