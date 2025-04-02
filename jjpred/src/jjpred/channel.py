"""Channels are the various locations where J&J products are sold."""

from __future__ import annotations

from collections import UserList
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import auto
import re
from typing import (
    Any,
    NamedTuple,
    Protocol,
    TypedDict,
    cast,
    runtime_checkable,
)
import polars as pl
from jjpred.countryflags import CountryFlags
from jjpred.parse.patternmatch import (
    CompiledMatchSkip,
    DictionaryMatcher,
    ReMatchCondition,
    ReMatchResult,
    ReMatcher,
    StringPattern,
)
from jjpred.structlike import FieldMeta, MemberType, StructLike
from jjpred.utils.multidict import MultiDict
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
    """No mode information."""
    ALL_MODE = auto()
    """All mode information was blurred."""


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

    country_info: PossibleCountries | FixedCountries
    distribution_mode: DistributionMode


class Platform(PlatformAttrs, EnumLike):
    """A platform represents a group of related channels. For example: "Amazon"
    or "HongMall"."""

    Amazon = (
        PossibleCountries([x for x in CountryFlags]),
        DistributionMode.RETAIL,
    )
    Faire = (
        FixedCountries(CountryFlags.US),
        DistributionMode.WHOLESALE,
    )
    HongMall = (
        PossibleCountries([CountryFlags.CA, CountryFlags.US]),
        DistributionMode.RETAIL,
    )
    JanAndJul = (
        FixedCountries(CountryFlags.CA | CountryFlags.US),
        DistributionMode.RETAIL,
    )
    PopUp = (
        FixedCountries(CountryFlags.CA),
        DistributionMode.RETAIL,
    )
    Bay = (
        FixedCountries(CountryFlags.CA),
        DistributionMode.RETAIL,
    )
    Walmart = (
        PossibleCountries([CountryFlags.CA, CountryFlags.US]),
        DistributionMode.RETAIL,
    )
    Warehouse = (
        PossibleCountries([CountryFlags.CA, CountryFlags.CN]),
        DistributionMode.WAREHOUSE,
    )
    Wholesale = (
        FixedCountries(CountryFlags.all_regions()),
        DistributionMode.WHOLESALE,
    )
    XiaoHongShu = (
        PossibleCountries([CountryFlags.CA, CountryFlags.US, CountryFlags.CN]),
        DistributionMode.RETAIL,
    )
    AllChannels = (
        FixedCountries(CountryFlags.all_regions() | CountryFlags.GlobalUS),
        DistributionMode.NO_MODE,
    )

    def __str__(self) -> str:
        return self.name


def platform_matcher(
    cls,
) -> ReMatcher:
    """Creates matcher to parse a string that might represent any channel."""
    return ReMatcher(
        "all_channels",
        [x.channel_matcher.match_skip for x in cls],
        ReMatchCondition.DeepAny,
    )


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


class ChannelDictType(TypedDict):
    platform: Platform
    country_flag: CountryFlags
    mode: DistributionMode


@dataclass
class RawChannel:
    platform: Platform
    country_flag: CountryFlags
    mode: DistributionMode

    def as_dict(self) -> ChannelDictType:
        return {
            "platform": self.platform,
            "country_flag": self.country_flag,
            "mode": self.mode,
        }


# amazon_matcher asked to match: Inv_AMZ USA
# amazon_matcher asked to match: Inv_AMZ CA
# amazon_matcher asked to match: Inv_AMZ MX
# amazon_matcher asked to match: Inv_AMZ AU
# amazon_matcher asked to match: Inv_AMZ UK
# amazon_matcher asked to match: Inv_AMZ EU


def create_amazon(country_flag: CountryFlags) -> ChannelDictType:
    assert PossibleCountries([x for x in CountryFlags]).contains_any(
        country_flag
    )
    return RawChannel(
        Platform.Amazon, country_flag, DistributionMode.RETAIL
    ).as_dict()


def create_janandjul(
    country_flag: CountryFlags | None = None,
) -> ChannelDictType:
    country_info = FixedCountries(CountryFlags.CA | CountryFlags.US)
    if country_flag is not None:
        assert country_info.contains_any(country_flag)
    else:
        country_flag = country_info.fixed

    return RawChannel(
        Platform.Amazon, country_flag, DistributionMode.RETAIL
    ).as_dict()


KNOWN_CHANNEL_MATCHERS: dict[str, ChannelDictType] = MultiDict(
    {
        ("amazon ca", "amazon.ca", "inv_amz ca"): create_amazon(
            CountryFlags.CA
        ),
        ("amazon us", "amazon.com", "inv_amz usa"): create_amazon(
            CountryFlags.US
        ),
        ("amazon mx", "amazon.com.mx", "inv_amz mx"): create_amazon(
            CountryFlags.MX
        ),
        ("amazon au", "amazon.com.au", "inv_amz au"): create_amazon(
            CountryFlags.AU
        ),
        (
            "amazon uk",
            "amazon.co.uk",
            "amazon.uk",
            "inv_amz uk",
        ): create_amazon(CountryFlags.UK),
        (
            "amazon de",
            "amazon eu",
            "amazon.eu",
            "amazon.de",
            "inv_amz eu",
        ): create_amazon(CountryFlags.DE),
        ("amazon global us",): create_amazon(CountryFlags.GlobalUS),
        ("thebay.ca",): RawChannel(
            Platform.Bay, CountryFlags.CA, DistributionMode.RETAIL
        ).as_dict(),
        ("faire.com",): RawChannel(
            Platform.Faire,
            CountryFlags.US,
            DistributionMode.WAREHOUSE,
        ).as_dict(),
        ("hongmall.ca",): RawChannel(
            Platform.HongMall,
            CountryFlags.CA,
            DistributionMode.RETAIL,
        ).as_dict(),
        ("hongmall.com",): RawChannel(
            Platform.HongMall,
            CountryFlags.US,
            DistributionMode.RETAIL,
        ).as_dict(),
        ("janandjul.com",): create_janandjul(),
        (
            "pop-up shop",
            "vancouver showroom",
            "surrey showroom",
            "richmond showroom",
        ): RawChannel(
            Platform.PopUp,
            CountryFlags.CA,
            DistributionMode.RETAIL,
        ).as_dict(),
        ("walmart.ca",): RawChannel(
            Platform.Walmart,
            CountryFlags.CA,
            DistributionMode.WAREHOUSE,
        ).as_dict(),
        ("walmart.com",): RawChannel(
            Platform.Walmart,
            CountryFlags.US,
            DistributionMode.WAREHOUSE,
        ).as_dict(),
        ("warehouse ca",): RawChannel(
            Platform.Warehouse, CountryFlags.CA, DistributionMode.WAREHOUSE
        ).as_dict(),
        ("wholesale",): RawChannel(
            Platform.Wholesale,
            CountryFlags.all_regions(),
            DistributionMode.WAREHOUSE,
        ).as_dict(),
        ("xiaohongshu.ca",): RawChannel(
            Platform.XiaoHongShu,
            CountryFlags.CA,
            DistributionMode.RETAIL,
        ).as_dict(),
        ("xiaohongshu.cn",): RawChannel(
            Platform.XiaoHongShu,
            CountryFlags.CN,
            DistributionMode.RETAIL,
        ).as_dict(),
        ("xiaohongshu.us",): RawChannel(
            Platform.XiaoHongShu,
            CountryFlags.US,
            DistributionMode.RETAIL,
        ).as_dict(),
    }
).as_dict()

# {

#     # f"{x.name} {y} {x.distribution_mode}": {
#     #     "platform": x,
#     #     "country_flag": x.country_info,
#     #     "mode": x.distribution_mode,
#     # }
#     # for x in Platform
#     # for y in x.country_info
# }

KNOWN_AMAZON_CHANNEL_MATCHERS = {
    k: v
    for k, v in KNOWN_CHANNEL_MATCHERS.items()
    if v["platform"] == Platform.Amazon
}


class Channel(
    StructLike,
    matcher=DictionaryMatcher("all_channels", KNOWN_CHANNEL_MATCHERS),
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
    def from_dict(
        cls, x: Mapping[str, str | Platform | CountryFlags | DistributionMode]
    ) -> Channel:
        platform = Platform.try_from(x["platform"])
        if platform:
            country = CountryFlags.try_from(x.get("country_flag"))
            if (country is None or country == CountryFlags(0)) and isinstance(
                platform.country_info, FixedCountries
            ):
                country = platform.country_info.fixed
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
            country_flag = "COUNTRY=??"
        elif country_flag == CountryFlags.all_regions():
            country_flag = "ALL_REGION"

        mode = self.mode.try_to_string()
        if mode is None:
            mode = "MODE=??"

        return f"{self.platform} {country_flag} {mode}"

    @classmethod
    def map_polars_struct_to_string(cls, polars_struct: Any) -> str:
        assert all([x in polars_struct.keys() for x in Channel.members()])
        return Channel.from_dict(polars_struct).pretty_string_repr()


# class Platform(PlatformAttrs, EnumLike):
#     """A platform represents a group of related channels. For example: "Amazon"
#     or "HongMall"."""

#     Amazon = (
#         ChannelMatcher(
#             ["amazon", "amz"],
#             PossibleCountries([x for x in CountryFlags]),
#         ),
#         DistributionMode.RETAIL,
#     )
#     Faire = (
#         ChannelMatcher(["faire"], FixedCountries(CountryFlags.US)),
#         DistributionMode.WHOLESALE,
#     )
#     HongMall = (
#         ChannelMatcher(
#             ["hongmall"],
#             PossibleCountries([CountryFlags.CA, CountryFlags.US]),
#         ),
#         DistributionMode.RETAIL,
#     )
#     JanAndJul = (
#         ChannelMatcher(
#             ["janandjul"],
#             FixedCountries(CountryFlags.CA | CountryFlags.US),
#             skip_patterns=[r"^inv_jj"],
#         ),
#         DistributionMode.RETAIL,
#     )
#     PopUp = (
#         ChannelMatcher(
#             [
#                 "pop-up shop",
#                 "popup",
#                 "vancouver showroom",
#             ],
#             FixedCountries(CountryFlags.CA),
#         ),
#         DistributionMode.RETAIL,
#     )
#     Bay = (
#         ChannelMatcher(["thebay", "thebay"], FixedCountries(CountryFlags.CA)),
#         DistributionMode.RETAIL,
#     )
#     Walmart = (
#         ChannelMatcher(
#             ["walmart"],
#             PossibleCountries([CountryFlags.CA, CountryFlags.US]),
#         ),
#         DistributionMode.RETAIL,
#     )
#     Warehouse = (
#         ChannelMatcher(
#             ["inv_wh", "warehouse"],
#             PossibleCountries([CountryFlags.CA, CountryFlags.CN]),
#         ),
#         DistributionMode.WAREHOUSE,
#     )
#     Wholesale = (
#         ChannelMatcher(
#             ["wholesale"],
#             FixedCountries(CountryFlags.all_regions()),
#         ),
#         DistributionMode.WHOLESALE,
#     )
#     XiaoHongShu = (
#         ChannelMatcher(
#             ["xiaohongshu"],
#             PossibleCountries(
#                 [CountryFlags.CA, CountryFlags.US, CountryFlags.CN]
#             ),
#         ),
#         DistributionMode.RETAIL,
#     )
#     AllChannels = (
#         ChannelMatcher(
#             ["all channels"],
#             FixedCountries(CountryFlags.all_regions() | CountryFlags.GlobalUS),
#         ),
#         DistributionMode.NO_MODE,
#     )

#     @classmethod
#     def matcher(
#         cls,
#     ) -> ReMatcher:
#         """Creates matcher to parse a string that might represent any channel."""
#         return ReMatcher(
#             "all_channels",
#             [x.channel_matcher.match_skip for x in cls],
#             ReMatchCondition.DeepAny,
#         )

#     @classmethod
#     def amazon_matcher(
#         cls,
#     ) -> ReMatcher:
#         """Creates a matcher to parse a string that might represent an Amazon channel."""
#         return ReMatcher(
#             "amazon_matcher",
#             [
#                 x.channel_matcher.match_skip
#                 for x in cls
#                 if "amazon" in x.name.lower()
#             ],
#             ReMatchCondition.DeepAny,
#         )

#     @classmethod
#     def from_str(cls, string: str) -> Self:
#         low_string = string.lower()
#         for x in cls:
#             if x.name.lower() == low_string or x.channel_matcher.apply(string):
#                 return x

#         raise ValueError(f"Cannot parse {string} as {cls.__qualname__}.")

#     def __str__(self) -> str:
#         return self.name
