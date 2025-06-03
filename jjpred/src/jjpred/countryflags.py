"""A J&J channel can operate in multiple countries."""

from __future__ import annotations

from enum import auto

from jjpred.parse.patternmatch import StringPattern
from jjpred.utils.polars import IntFlagLike


def regex_domain(domains: list[str]) -> StringPattern:
    """Create a regex expression for capturing the domain part of a URL-like
    representation of a country (for example, in "Amazon.ca", the ".ca" is the
    channel's country representation.)"""
    return StringPattern().any_of(
        [
            StringPattern(x).fragmentlike(default_start_patterns=[])
            for x in domains
        ]
    )


def country_string_pattern(fragments: list[str]) -> StringPattern:
    """Create a regex expression for matching against any one of the string
    fragments that represent's a country."""
    return StringPattern().any_of(*fragments)


class CountryFlags(IntFlagLike):
    """A J&J channel can operate in multiple countries. So we use an ``IntFlag``
    to represent the countries it is operating in."""

    AU = auto()
    """Australia"""
    CA = auto()
    """Canada"""
    CN = auto()
    """China"""
    DE = auto()
    """Germany"""
    EU = auto()
    """European Union"""
    GlobalUS = auto()
    """Amazon Global US"""
    MX = auto()
    """Mexico"""
    UK = auto()
    """United Kingdom"""
    US = auto()
    """United States"""
    JP = auto()
    """Japan"""

    def try_to_string(self) -> str | None:
        if self == CountryFlags.all_regions():
            return "ALL_REGION"
        else:
            return self.name

    @classmethod
    def all_regions(cls) -> CountryFlags:
        country_flag = CountryFlags(0)
        for x in CountryFlags:
            if x != CountryFlags.GlobalUS:
                country_flag |= x

        return country_flag

    def string_pattern(self) -> StringPattern:
        """The string pattern (regular expression) used to match for a
        particular country."""
        patterns: list[StringPattern] = []
        name = str(self)
        for country in self:
            match country:
                case CountryFlags.AU:
                    patterns.append(
                        country_string_pattern(["au", "australia", r"com\.au"])
                    )
                case CountryFlags.CA:
                    patterns.append(
                        country_string_pattern(["ca", "can", "canada", "jj"])
                    )
                case CountryFlags.CN:
                    patterns.append(
                        country_string_pattern(["cn", "china", "total cn"])
                    )
                case CountryFlags.DE:
                    patterns.append(
                        country_string_pattern(["eu", "de", "germany"])
                    )
                # case CountryFlags.EU:
                #     patterns.append(country_string_pattern(["eu"]))
                case CountryFlags.MX:
                    patterns.append(
                        country_string_pattern(["mx", "mexico", r"com\.mx"])
                    )
                case CountryFlags.UK:
                    patterns.append(
                        country_string_pattern(["uk", "gb", r"co\.uk"])
                    )
                case CountryFlags.US:
                    patterns.append(
                        country_string_pattern(
                            ["us", "usa", "america", r"com$"]
                        )
                    )
        return StringPattern().any_of(*patterns).named(f"country_{name}")
