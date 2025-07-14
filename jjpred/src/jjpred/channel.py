"""Channels are the various locations where J&J products are sold."""

from __future__ import annotations

from collections import UserList
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import auto, unique
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


class SubCountry(EnumLike):
    """:py:class:`EnumLike` representing different areas within a country/region
    that a channel might operate in."""

    ALL = auto()
    """The default value: the entire country."""
    EAST = auto()
    """The EAST part of a country."""
    WEST = auto()
    """The WEST part of a country."""


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
class PossibleSubCountry(UserList[SubCountry]):
    data: list[SubCountry]


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

    ix: int
    country_info: PossibleCountries | FixedCountries
    distribution_mode: DistributionMode
    sub_country_info: PossibleSubCountry


@unique
class Platform(PlatformAttrs, EnumLike):
    """A platform represents a group of related channels. For example: "Amazon"
    or "HongMall"."""

    Amazon = (
        auto(),
        PossibleCountries(
            [x for x in CountryFlags if x != CountryFlags.GlobalUS]
        ),
        DistributionMode.RETAIL,
        PossibleSubCountry([SubCountry.ALL]),
    )
    Faire = (
        auto(),
        FixedCountries(CountryFlags.US),
        DistributionMode.WHOLESALE,
        PossibleSubCountry([SubCountry.ALL]),
    )
    HongMall = (
        auto(),
        PossibleCountries([CountryFlags.CA, CountryFlags.US]),
        DistributionMode.RETAIL,
        PossibleSubCountry([SubCountry.ALL]),
    )
    JJWeb = (
        auto(),
        FixedCountries(CountryFlags.CA | CountryFlags.US),
        DistributionMode.RETAIL,
        PossibleSubCountry([SubCountry.ALL, SubCountry.EAST, SubCountry.WEST]),
    )
    JJPhysical = (
        auto(),
        FixedCountries(CountryFlags.CA),
        DistributionMode.RETAIL,
        PossibleSubCountry([SubCountry.ALL, SubCountry.EAST, SubCountry.WEST]),
    )
    Bay = (
        auto(),
        FixedCountries(CountryFlags.CA),
        DistributionMode.RETAIL,
        PossibleSubCountry([SubCountry.ALL]),
    )
    Walmart = (
        auto(),
        PossibleCountries([CountryFlags.CA, CountryFlags.US]),
        DistributionMode.RETAIL,
        PossibleSubCountry([SubCountry.ALL]),
    )
    Warehouse = (
        auto(),
        PossibleCountries([CountryFlags.CA, CountryFlags.CN]),
        DistributionMode.WAREHOUSE,
        PossibleSubCountry([SubCountry.ALL]),
    )
    Wholesale = (
        auto(),
        FixedCountries(CountryFlags.all_regions()),
        DistributionMode.WHOLESALE,
        PossibleSubCountry([SubCountry.ALL]),
    )
    XiaoHongShu = (
        auto(),
        PossibleCountries([CountryFlags.CA, CountryFlags.US, CountryFlags.CN]),
        DistributionMode.RETAIL,
        PossibleSubCountry([SubCountry.ALL]),
    )
    AllChannels = (
        auto(),
        FixedCountries(CountryFlags.all_regions() | CountryFlags.GlobalUS),
        DistributionMode.NO_MODE,
        PossibleSubCountry([SubCountry.ALL]),
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
    """Typed dictionary useful for making sure that we are returning all known
    aspecets of a channel."""

    platform: Platform
    country_flag: CountryFlags
    mode: DistributionMode
    sub_country: SubCountry


@dataclass
class RawChannel:
    platform: Platform
    country_flag: CountryFlags
    mode: DistributionMode
    sub_country: SubCountry = field(default=SubCountry.ALL)

    def as_dict(self) -> ChannelDictType:
        return {
            "platform": self.platform,
            "country_flag": self.country_flag,
            "mode": self.mode,
            "sub_country": self.sub_country,
        }


def create_amazon(country_flag: CountryFlags) -> ChannelDictType:
    assert PossibleCountries([x for x in CountryFlags]).contains_any(
        country_flag
    )
    return RawChannel(
        Platform.Amazon,
        country_flag
        if country_flag != CountryFlags.GlobalUS
        else CountryFlags(0),
        DistributionMode.RETAIL,
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
        Platform.JJWeb,
        country_flag,
        DistributionMode.RETAIL,
    ).as_dict()


KNOWN_CHANNEL_MATCHERS: dict[str, ChannelDictType] = MultiDict(
    {
        ("amazon ca", "amazon.ca", "inv_amz ca"): create_amazon(
            CountryFlags.CA
        ),
        ("amazon us", "amazon.com", "inv_amz usa"): create_amazon(
            CountryFlags.US
        ),
        (
            "amazon mx",
            "amazon.com.mx",
            "amazon.mx",
            "inv_amz mx",
        ): create_amazon(CountryFlags.MX),
        (
            "amazon au",
            "amazon.com.au",
            "amazon.au",
            "inv_amz au",
        ): create_amazon(CountryFlags.AU),
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
        (
            "amazon jp",
            "amazon.co.jp",
            "amazon.jp",
            "inv_amz jp",
        ): create_amazon(CountryFlags.JP),
        ("amazon global us",): create_amazon(CountryFlags.GlobalUS),
        ("thebay.ca",): RawChannel(
            Platform.Bay, CountryFlags.CA, DistributionMode.RETAIL
        ).as_dict(),
        ("faire.com", "wholesale-faire.com"): RawChannel(
            Platform.Faire,
            CountryFlags.US,
            DistributionMode.WHOLESALE,
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
        ("janandjul.com", "jjweb", "janandjul.com (all)"): RawChannel(
            Platform.JJWeb,
            CountryFlags.CA | CountryFlags.US,
            DistributionMode.RETAIL,
            sub_country=SubCountry.ALL,
        ).as_dict(),
        (
            "usa.janandjul.com",
            "jjweb usa",
            "jjweb us",
            "janandjul.com (us)",
        ): RawChannel(
            Platform.JJWeb,
            CountryFlags.US,
            DistributionMode.RETAIL,
            sub_country=SubCountry.ALL,
        ).as_dict(),
        (
            "ca.janandjul.com",
            "janandjul.com (ca)",
        ): RawChannel(
            Platform.JJWeb,
            CountryFlags.CA,
            DistributionMode.RETAIL,
            sub_country=SubCountry.ALL,
        ).as_dict(),
        (
            "ca.janandjul.com (east)",
            "ca.janandjul.com east",
            "jjweb ca east",
        ): RawChannel(
            Platform.JJWeb,
            CountryFlags.CA,
            DistributionMode.RETAIL,
            sub_country=SubCountry.EAST,
        ).as_dict(),
        (
            "ca.janandjul.com (west)",
            "ca.janandjul.com west",
            "jjweb ca west",
        ): RawChannel(
            Platform.JJWeb,
            CountryFlags.CA,
            DistributionMode.RETAIL,
            sub_country=SubCountry.WEST,
        ).as_dict(),
        (
            "pop-up shop",
            "vancouver showroom",
            "surrey showroom",
            "richmond showroom",
            "surrey shop",
            "richmond shop",
        ): RawChannel(
            Platform.JJPhysical,
            CountryFlags.CA,
            DistributionMode.RETAIL,
        ).as_dict(),
        ("walmart.ca",): RawChannel(
            Platform.Walmart,
            CountryFlags.CA,
            DistributionMode.RETAIL,
        ).as_dict(),
        ("walmart.com",): RawChannel(
            Platform.Walmart,
            CountryFlags.US,
            DistributionMode.RETAIL,
        ).as_dict(),
        ("warehouse ca", "jj warehouse", "wh surrey", "wh-surrey"): RawChannel(
            Platform.Warehouse, CountryFlags.CA, DistributionMode.WAREHOUSE
        ).as_dict(),
        ("wh china",): RawChannel(
            Platform.Warehouse, CountryFlags.CN, DistributionMode.WAREHOUSE
        ).as_dict(),
        (
            "unknown warehouse",
            "wh richmond",
        ): RawChannel(
            Platform.Warehouse, CountryFlags(0), DistributionMode.WAREHOUSE
        ).as_dict(),
        ("wholesale",): RawChannel(
            Platform.Wholesale,
            CountryFlags.all_regions(),
            DistributionMode.WHOLESALE,
        ).as_dict(),
        ("wholesale-ca", "wholesale ca"): RawChannel(
            Platform.Wholesale,
            CountryFlags.CA,
            DistributionMode.WHOLESALE,
        ).as_dict(),
        ("wholesale-us", "wholesale us"): RawChannel(
            Platform.Wholesale,
            CountryFlags.US,
            DistributionMode.WHOLESALE,
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

    sub_country: SubCountry = field(
        default=SubCountry.ALL,
        kw_only=True,
        hash=False,
        compare=True,
        metadata=FieldMeta(
            MemberType.PRIMARY,
            str,
            SubCountry.polars_type(),
            intermediate_polars_dtype=pl.String(),
        ),
    )
    """The sub-country that this channel operates in (e.g. EAST or WEST, usually
    ALL)."""

    @classmethod
    def from_dict(
        cls,
        x: Mapping[
            str, str | Platform | CountryFlags | DistributionMode | SubCountry
        ],
    ) -> Channel:
        platform = Platform.try_from(x["platform"])
        if platform:
            country = CountryFlags.try_from(x.get("country_flag"))
            if (country is None or country == CountryFlags(0)) and isinstance(
                platform.country_info, FixedCountries
            ):
                country = platform.country_info.fixed

            mode = DistributionMode.try_from(x.get("mode"))
            if mode is not None:
                assert mode == platform.distribution_mode, (
                    f"when parsing: {x}"
                    f"{mode=} != {platform.distribution_mode=}"
                )

            sub_country = SubCountry.try_from(x.get("sub_country"))
            if sub_country is not None:
                assert sub_country in platform.sub_country_info, (
                    f"{sub_country=} not in {platform.sub_country_info=}"
                )

            channel_dict: dict[str, Any] = {
                "channel": x.get("channel"),
                "platform": platform.name,
                "country_flag": country,
                "mode": mode,
                "sub_country": sub_country,
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

        sub_country = self.sub_country.try_to_string()
        if sub_country is None or sub_country.lower() == "all":
            sub_country = ""

        return " ".join(
            x
            for x in [self.platform, country_flag, sub_country, mode]
            if len(x) > 0
        )

    @classmethod
    def map_polars_struct_to_string(cls, polars_struct: Any) -> str:
        assert all([x in polars_struct.keys() for x in Channel.members()])
        return Channel.from_dict(polars_struct).pretty_string_repr()
