"""Utilities for dealing with and manipulating dates."""

from dataclasses import dataclass
import locale
from enum import Enum, auto
from functools import total_ordering
from typing import NamedTuple, Self
import datetime as dt
import polars as pl
from jjpred.utils.polars import scalar_as_series


type Month = int
"""Months are represented by integers."""


class YearMonthDay(NamedTuple):
    """Helps keep track of year/month/day information."""

    year: int
    month: int
    day: int


type DateLike = str | dt.date | YearMonthDay | Date
"""Enumerates types that are considered to be date-like."""


def date_as_series(date: DateLike) -> pl.Series:
    """Convert a date-like into a series of dates."""
    return scalar_as_series(
        date, converter=lambda x: Date.from_datelike(x).date, dtype=pl.Date()
    )


# TODO: use babel to manage locale specific date formatting and parsing
def get_system_locale() -> str | None:
    """Check if the system's locale matches the expectation this program was
    built on.

    Unfortunately, Python's :py:func:`strftime` and :py:func:`strptime`
    functions are sensitive to the system locale. The system locale cannot be
    temporily spoofed for the running of our program, which means that our
    program will only reliably run on machines which are set to system locale
    EN/CA (although EN/US should also work, this has yet to be tested).

    In order to properly fix this problem, we should use the
    `babel <https://pypi.org/project/babel/>`_ package to parse and format
    dates, instead of Python's built in date formatters."""
    return locale.getlocale(locale.LC_CTYPE)[0]


def assert_assumed_locale(assumed_locale: list[str]):
    """Check if the locale is one of the valid locales (current EN/CA),
    otherwise raise an error to stop program execution."""
    system_locale = get_system_locale()
    if system_locale not in assumed_locale:
        raise RuntimeError(
            f"{system_locale=} not in {assumed_locale=}, which will cause issues"
            " for date formatting and parsing."
        )


ASSUMED_LOCALE = ["en_CA", "English_Canada"]
"""Unfortunately, Python's :py:func:`strftime` and :py:func:`strptime`
    functions are sensitive to the system locale. The system locale cannot be
    temporily spoofed for the running of our program, which means that our
    program will only reliably run on machines which are set to system locale
    EN/CA (although EN/US should also work, this has yet to be tested).

    In order to properly fix this problem, we should use the
    `babel <https://pypi.org/project/babel/>`_ package to parse and format
    dates, instead of Python's built in date formatters."""


@total_ordering
@dataclass
class Date:
    """An object representing a simple date (year, month, day), without any
    time."""

    year: int
    month: int
    day: int
    date: dt.date

    @classmethod
    def today(cls) -> Self:
        return cls.from_date(dt.datetime.today())

    def with_year(self, year: int) -> Self:
        return self.__class__.from_date(dt.date(year, self.month, self.day))

    def with_month(self, month: int) -> Self:
        return self.__class__.from_date(dt.date(self.year, month, self.day))

    def with_day(self, day: int) -> Self:
        return self.__class__.from_date(dt.date(self.year, self.month, day))

    def strftime(self, fmt: str) -> str:
        assert_assumed_locale(ASSUMED_LOCALE)
        return self.date.strftime(
            fmt,
        )

    def format_as(self, fmt: str) -> str:
        assert_assumed_locale(ASSUMED_LOCALE)
        return self.strftime(fmt)

    @classmethod
    def from_datelike(cls, x: DateLike) -> Self:
        assert_assumed_locale(ASSUMED_LOCALE)
        if isinstance(x, str):
            return cls.from_str(x)
        elif isinstance(x, dt.date):
            return cls.from_date(x)
        elif isinstance(x, cls):
            return x
        elif isinstance(x, YearMonthDay):
            return cls(
                x.year,
                x.month,
                x.day,
                dt.date(x.year, x.month, x.day),
            )
        else:
            raise ValueError(f"Cannot convert {x} to SimpleDate")

    @classmethod
    def from_date(cls, x: dt.date) -> Self:
        return cls(x.year, x.month, x.day, x)

    @classmethod
    def from_ymd(cls, year: int, month: Month, day: int) -> Self:
        return cls.from_date(dt.date(year, month, day))

    @classmethod
    def strptime(cls, date_string: str, fmt_str: str) -> Self:
        assert_assumed_locale(ASSUMED_LOCALE)
        return cls.from_date(dt.datetime.strptime(date_string, fmt_str).date())

    @classmethod
    def from_str(cls, x: str) -> Self:
        assert_assumed_locale(ASSUMED_LOCALE)
        date_formats = [
            "%Y-%b-%d",
            "%Y-%m-%d",
            "%b-%Y",
            "%Y-%b",
            "%Y %b",
            "%b %Y",
        ]
        exception_history = []
        for date_format in date_formats:
            try:
                return cls.strptime(x, date_format)
            except ValueError as e:
                exception_history.append(e)

        raise ExceptionGroup(
            f"Could not parse {x} as one of {date_formats}", exception_history
        )

    def fmt_default(self) -> str:
        """Format a date in 'default style' e.g. ``2024-09-09 -> 2024-SEP-09``"""
        return self.strftime(r"%Y-%b-%d").upper()
        # return f"{self.year}-{cal.month_abbr[self.month].upper()}-{pad_double_digit(self.day)}"

    def fmt_flat(self) -> str:
        """Format a date in 'flat style', e.g. ``2024-SEP-09 -> 20240909``"""
        return self.strftime(r"%Y%m%d")
        # m = pad_double_digit(self.month)
        # d = pad_double_digit(self.day)
        # return f"{self.year}{m}{d}"

    @classmethod
    def from_excel_file_path(cls, file: str) -> Self:
        assert_assumed_locale(ASSUMED_LOCALE)
        date_part = file.split(".")[0].split(" ")[-1]
        print(date_part)
        year = int(date_part[:4])
        month = int(date_part[4:6])
        day = int(date_part[6:])

        return cls.from_datelike(YearMonthDay(year, month, day))

    def as_polars_date(self) -> pl.Expr:
        return pl.date(self.year, self.month, self.day)

    def __str__(self) -> str:
        return self.fmt_default()

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other: Self) -> bool:
        return self.date < other.date

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.date == other.date
        raise ValueError(f"Cannot compare SimpleDate with {other}")

    def __sub__(self, other: Self) -> dt.timedelta:
        return (date_as_series(self.date) - date_as_series(other.date))[0]


class DateUnit(Enum):
    """:py:class:`Enum` enumerating the different date-units that are often
    useful in our for offsetting dates.

    This is geared towards using :py:func:`pl.Expr.dt.offset_by`.
    """

    YEAR = auto()
    """A full year."""
    QUARTER = auto()
    """A quarter of a year."""
    MONTH = auto()
    """A month."""
    WEEK = auto()
    """A week."""
    DAY = auto()
    """A day."""

    def __str__(self) -> str:
        match self:
            case DateUnit.MONTH:
                return "mo"
            case _:
                return self.name[0].lower()


@dataclass
class DateOffset:
    """A date offset: consists of a size and a unit."""

    size: int
    unit: DateUnit

    def __str__(self) -> str:
        return f"{self.size}{str(self.unit)}"

    def apply_to(self, start_date: DateLike) -> Date:
        """Offset given date by the given offset (to offset backwards, use a date
        offset with a negative size)."""
        series = date_as_series(start_date)
        return Date.from_datelike(series.dt.offset_by(str(self))[0])


def offset_date(date: DateLike, offset: DateOffset) -> Date:
    """Offset given date by the given offset (to offset backwards, use a date
    offset with a negative size)."""
    return offset.apply_to(date)


def today() -> Date:
    """Get today's date."""
    return Date.from_date(dt.datetime.today())


def first_day_next_month(date: DateLike) -> Date:
    """Get the first day of the next month relative to the given date."""
    return Date.from_datelike(
        date_as_series(date).dt.offset_by("1mo").dt.month_start()[0]
    )


def first_day(date: DateLike) -> Date:
    """Get the first day of the given date's month."""
    return Date.from_datelike(date_as_series(date).dt.month_start()[0])


def first_day_last_month(date: DateLike) -> Date:
    """Get the first day of the last month relative to the given date."""
    return Date.from_datelike(
        date_as_series(date).dt.offset_by("-1mo").dt.month_start()[0]
    )
