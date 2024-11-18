"""Parsing strings using regular expressions is a common task, especially when
reading input data."""

from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass
from enum import Enum, auto
from itertools import chain
import re
from typing import (
    Any,
    Protocol,
    Self,
    cast,
    runtime_checkable,
)
from collections.abc import Mapping

from jjpred.utils.typ import (
    Additive,
    ScalarOrList,
    as_list,
)


class PatternMatchResult(UserDict[str, Any]):
    """Stores the result of a pattern match operation"""

    data: dict[str, Any]

    def update(self, other: Mapping[str, Any], **kwargs: Mapping[str, Any]):  # type: ignore
        for k, v in chain(other.items(), kwargs.items()):
            if v is not None:
                self[k] = v


class ReMatchResult(PatternMatchResult):
    """A pattern match result for a regex based matcher."""

    def __init__(self, data: dict[str, str | Any]):
        self.data = {}
        self.update(data)

    @classmethod
    def from_re_match(
        cls,
        match: re.Match[str] | None,
    ) -> Self | None:
        if match:
            return cls(match.groupdict())


class FindMode(Enum):
    """:py:class:`Enum` enumerating modes that can be used to find a pattern in
    a string."""

    Iter = auto()
    """Match using `finditer`_. Get an iterator of the non-overlapping matches
    of a pattern within the string.

    .. _finditer: https://docs.python.org/3/library/re.html#re.finditer
    """
    Full = auto()
    """Match using `fullmatch`_. Find all matches within a string.

    .. _fullmatch: https://docs.python.org/3/library/re.html#re.fullmatch
    """
    Standard = auto()
    """Match using `match`_. Note: this only attempts to find a match starting
    from the beginning of the string.

    .. _match: https://docs.python.org/3/library/re.html#re.match
    """


class PatternGroup(str, Enum):
    """:py:class:`Enum` enumerating the kinds of string-matching (RegEx)
    pattern groups that can be built."""

    Named = f"(?P<{"{name}"}>{{pattern}})"
    """A named RegEx capture group."""
    Anonymous = "({{pattern}})"
    """An anonymous RegEx capture group (will be 'named' by a number)."""
    NoCapture = "(?:{{pattern}})"
    """A pattern that will not be captured."""


class StringPattern:
    """A string (uncompiled) representation of a regex pattern."""

    pattern: str
    """A string (uncompiled) representation of a regex pattern.
    """
    _grouping: PatternGroup | None = None
    """Grouping to apply to on pattern compilation."""
    _grouped: bool = False
    """Indicates whether this pattern is grouped."""

    def __init__(
        self, pattern: str | Self | None = None, _grouped: bool = False
    ) -> None:
        """Create a string pattern. Use one of the dedicated ``create_*``
        methods on this class to create a pattern manually.

        :param pattern: Optional string (uncompiled) representation of a regex
            pattern, or another string pattern object. If ``None``, then an
            empty pattern ``""`` is used.
        """
        if pattern is not None:
            if isinstance(pattern, str):
                self.pattern = pattern
            else:
                self.pattern = pattern.pattern
        else:
            self.pattern = ""

        self._grouped = _grouped

    def fmt(self, format: str, _grouped: bool = False) -> Self:
        if not self.is_empty():
            return self.__class__(
                format.format(pattern=self.pattern), _grouped=_grouped
            )
        else:
            return self

    def named(
        self,
        name: str,
    ) -> Self:
        """Generate a named capture group version of this pattern.

        :param name: Name of the capture group.
        :return: A named capture group version of this pattern
        """
        return self.fmt(f"(?P<{name}>" + "{pattern})", _grouped=True)

    def capture(self, name: str | None = None):
        """Generate a `capturing group` version of this pattern.

        :param name: Optional name to create a named capture group, otherwise
            create an anonymous capture group.
        :return: A named/anonymous capture group version of this pattern.
        """
        group_meta = ""
        if name:
            group_meta = f"P<{name}>"
        return self.fmt(group_meta + "{pattern}", _grouped=True)

    def zero_or_more(self) -> Self:
        """Create version of this pattern that will be found zero or more times,
        corresponds to regex `star <https://regex101.com/r/U2Ew3b/1>`_.

        :return:
        """
        return self.fmt("{pattern}*")

    def no_capture(self) -> Self:
        """Create a `non-capturing group <https://regex101.com/r/o1lOE0/1>`_
        version of this pattern.

        :return:
        """
        if self._grouped:
            return self
        else:
            return self.fmt(r"(?:{pattern})", _grouped=True)

    def optional(self) -> Self:
        """Create an `optional <https://regex101.com/r/1fiVbH/1>`_
        version of this pattern.

        :return:
        """
        return self.fmt(r"{pattern}?")

    def is_empty(self) -> bool:
        """Check if the pattern is empty (``""``)."""
        return len(self.pattern) == 0

    def concatenate(
        self,
        *pattern_args: ScalarOrList[str | StringPattern],
        joiner: str | StringPattern = "",
    ) -> Self:
        """Concatenate the given patterns.

        :param joiner: String used to join patterns, defaults to ``""``.
        :return:
        """
        patterns = [
            str(x)
            for x in (
                [self]
                + [
                    StringPattern(y)
                    for xs in pattern_args
                    for y in as_list(xs)
                ]
            )
            if not x.is_empty()
        ]
        if isinstance(joiner, StringPattern):
            joiner = joiner.pattern
        return self.__class__(joiner.join(patterns))

    def any_of(
        self,
        *pattern_args: ScalarOrList[str | StringPattern],
    ) -> Self:
        """Create a pattern that matches
        `any of <https://regex101.com/r/HLCYcj/1>`_ the given options
        ("greedy" disjoint union).

        :return:
        """
        return self.concatenate(*pattern_args, joiner="|")

    def fragmentlike(
        self,
        start_patterns: list[str | StringPattern] = [],
        end_patterns: list[str | StringPattern] = [],
        default_start_patterns: list[str | StringPattern] = [
            r"^",
            r"\b",
        ],
        default_end_pattern: list[str | StringPattern] = [r"$", r"\b"],
    ) -> Self:
        """Create a version of this pattern that will be wrapped between the
        given start and end patterns: in other words, it is a "fragment" where
        the fragment boundaries are given by the start and end patterns.

        :param start_patterns: Patterns that mark the start of the fragment,
            defaults to empty list.
        :param end_patterns: Patterns that mark the end of the fragment,
            defaults to empty list.
        :param default_start_patterns: Patterns that will be applied by default,
            in addition to any given start patterns.
        :param default_end_patterns: Patterns that will be applied by default in
            addition to any given end patterns.
        :return:
        """
        start = (
            StringPattern()
            .any_of(start_patterns + default_start_patterns)
            .no_capture()
        )
        end = (
            StringPattern()
            .any_of(end_patterns + default_end_pattern)
            .no_capture()
        )

        return cast(
            Self,
            start.concatenate(
                self if self._grouped else self.no_capture(), end, joiner=""
            ).no_capture(),
        )

    def neg_lookahead(self, lookahead: str | Self) -> Self:
        """Create a pattern that will `abort
        matching <https://regex101.com/r/32K6GL/1>`_ if it is found.

        :return:
        """
        return self.fmt(f"(?!{lookahead})" + "{pattern}")

    def compile[T: ReMatchResult](
        self,
        result_type: type[T],
        find_mode: FindMode = FindMode.Iter,
        flags: re.RegexFlag = re.RegexFlag(0),
    ) -> CompiledPattern:
        """Compile the pattern.

        :param result_type: The type of match result that will be generated by
            this pattern upon application to a target string.
        :param find_mode: How this pattern should be found within a target
            string it is applied to.
        :param flags: Any `regex
            flags <https://docs.python.org/3/library/re.html#flags>`_ that the
            compiled flag will obey.
        :return:
        """
        return CompiledPattern(
            self, result_type, find_mode=find_mode, flags=flags
        )

    def __repr__(self) -> str:
        return self.pattern

    def __str__(self) -> str:
        return self.pattern


class CompiledPattern[T: ReMatchResult]:
    """A compiled regex pattern."""

    compiled_pattern: re.Pattern[str]
    """The compiled regex pattern."""
    string_pattern: StringPattern
    """The string that the compiled pattern was generated from."""
    find_mode: FindMode
    """How this pattern should be found in the target string it is applied
    to."""
    flags: re.RegexFlag
    """Any `regex
            flags <https://docs.python.org/3/library/re.html#flags>`_ that the
            compiled flag will obey."""
    result_type: type[T]
    """The type of match result that will be generated by this pattern upon
    application to a target string."""

    def __init__(
        self,
        pattern: str | StringPattern | re.Pattern[str],
        result_type: type[T],
        find_mode: FindMode = FindMode.Iter,
        flags: re.RegexFlag = re.RegexFlag(0),
    ) -> None:
        if isinstance(pattern, re.Pattern):
            string_pattern = StringPattern(pattern.pattern)
            flags = re.RegexFlag(pattern.flags) | flags
        else:
            string_pattern = StringPattern(pattern)

        self.string_pattern = string_pattern
        self.find_mode = find_mode
        self.flags = flags
        self.compiled_pattern = re.compile(str(pattern), flags=flags)
        self.result_type = result_type

    def string(self) -> str:
        """Get the string representation of this pattern."""
        return self.string_pattern.pattern

    def groupindex(self) -> Mapping[str, int]:
        """Get the
        ```groupindex`` <https://docs.python.org/3/library/re.html#re.Pattern.groupindex>`_
        of this pattern."""
        return self.compiled_pattern.groupindex

    def apply(self, target: str) -> T | None:
        """Apply this pattern to a target string."""
        target = target.strip()
        raw_match = None
        if self.find_mode == FindMode.Full:
            raw_match = self.compiled_pattern.fullmatch(target)
        else:
            if raw_match := self.compiled_pattern.match(target):
                pass
            elif self.find_mode == FindMode.Iter:
                matches = [
                    match for match in self.compiled_pattern.finditer(target)
                ]
                if len(matches) > 1:
                    # raise Exception(
                    #     f"No logic for handling multiple {matches=}"
                    # )
                    raw_match = matches[0]
                elif len(matches) == 1:
                    raw_match = matches[0]

        return self.result_type.from_re_match(raw_match)

    def recompile(
        self,
        string_pattern: StringPattern | None = None,
        find_mode: FindMode | None = None,
        flags: re.RegexFlag | None = None,
    ) -> Self:
        """Recompile this pattern with optional new settings; old settings will
        be re-used for those that are not newly provided."""
        if string_pattern is None:
            string_pattern = self.string_pattern

        if find_mode is None:
            find_mode = self.find_mode

        if flags is None:
            flags = self.flags

        return self.__class__(
            string_pattern, self.result_type, find_mode=find_mode, flags=flags
        )


class CompiledMatchSkip[T: ReMatchResult]:
    """Compiled version of a match-skip pattern.

    The skip pattern is checked for first in the target string. If it is found,
    then the match operation aborts. Otherwise, the match pattern is searched
    for in the target string.
    """

    match: CompiledPattern[T]
    """Pattern to be found within a target string."""
    skip: CompiledPattern[T] | None
    """First check to see if the skip pattern is found in the target string, if
    so, abort matching."""

    def __init__(
        self,
        match: CompiledPattern[T],
        skip: CompiledPattern[T] | None = None,
    ):
        self.match = match
        self.skip = skip

    def empty_result(self) -> T:
        return self.match.result_type({})

    def try_match(self, string: str) -> T | None:
        if not (self.skip and self.skip.apply(string)):
            return self.match.apply(string)

    def match_contains(self, string: str) -> bool:
        if string in self.get_pattern():
            return True
        else:
            return False

    def get_pattern(self) -> str:
        return self.match.string()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"match: {self.match.string()}, "
            f"skip: {self.skip.string() if self.skip is not None else None}"
            ")"
        )


class ReMatchCondition(Enum):
    """:py:class:`Enum` enumerating the different ways in which a list of regex
    patterns can be matched for within a target list of strings.

    If the condition is not met, matching is aborted."""

    DeepAll = auto()
    """Match all patterns, within all target strings."""
    DeepAny = auto()
    """Match any pattern, within all target strings."""
    WideAll = auto()
    """Match all patterns, within any one of the target strings."""


@runtime_checkable
class PatternMatcher(Additive, Protocol):
    """:py:class:`Protocol` for a pattern matching object."""

    name: str

    def apply(
        self, target_strings: ScalarOrList[str]
    ) -> PatternMatchResult | None:
        raise NotImplementedError()


@dataclass
class ReMatcher[T: ReMatchResult](PatternMatcher):
    """A regex-based pattern matcher."""

    name: str
    """The name of this pattern matcher object."""
    match_skips: list[CompiledMatchSkip[T]]
    """The match-skip objects defining this pattern matcher."""
    match_condition: ReMatchCondition

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"name: {self.name}, "
            f"match_skips: {self.match_skips.__repr__()}, "
            f"match_condition: {self.match_condition}"
            ")"
        )

    def __init__(
        self,
        name: str,
        match_skips: ScalarOrList[CompiledMatchSkip[T]],
        match_condition: ReMatchCondition,
    ) -> None:
        self.name = name
        self.match_skips = as_list(match_skips)
        self.match_condition = match_condition

    @classmethod
    def from_pattern(
        cls,
        name: str,
        pattern: CompiledPattern[T],
        mode: ReMatchCondition,
    ) -> Self:
        """Initialize a regex-based pattern matcher from a given compiled regex
        pattern."""
        return cls(name, CompiledMatchSkip(pattern), mode)

    def apply(self, target_strings: ScalarOrList[str]) -> T | None:
        """Apply this matcher to target strings."""
        match self.match_condition:
            case ReMatchCondition.WideAll:
                return self.__match_wide_all__(target_strings)
            case ReMatchCondition.DeepAll:
                return self.__match_deep_all__(target_strings)
            case ReMatchCondition.DeepAny:
                return self.__match_deep_any__(target_strings)
            case _:
                Exception(f"No logic to handle case {self.mode}")

    def __match_wide_all__(self, strings: ScalarOrList[str]) -> T | None:
        strings = as_list(strings)

        if len(strings) >= len(self.match_skips):
            if len(strings) > len(self.match_skips):
                strings = strings[len(strings) - len(self.match_skips) :]

            if len(self.match_skips) > 0:
                result = self.match_skips[0].empty_result()
                for match_and_skip, string in zip(
                    self.match_skips, strings, strict=True
                ):
                    if gd := match_and_skip.try_match(string):
                        result.update(gd)
                    else:
                        return None

                return result

    def __match_deep_all__(self, strings: ScalarOrList[str]) -> T | None:
        strings = as_list(strings)

        for string in strings:
            all_match = True
            result = None
            if len(self.match_skips) > 0:
                result = self.match_skips[0].empty_result()
                for match_skip in self.match_skips:
                    if gd := match_skip.try_match(string):
                        result.update(gd)
                    else:
                        all_match = False
                        break

            if all_match:
                return result

    def __match_deep_any__(self, strings: ScalarOrList[str]) -> T | None:
        strings = as_list(strings)

        for string in strings:
            for match_skip in self.match_skips:
                if gd := match_skip.try_match(string):
                    return gd

    def str_in_pattern(self, string: str) -> bool:
        """Check if the given string is in any one of the match parts of the
        match-skip patterns composing this pattern matcher."""
        return any([ms.match_contains(string) for ms in self.match_skips])

    @classmethod
    def combine(cls, left: Self, right: Self) -> Self:
        """Combine (concatenate) two pattern matchers in sequence.

        They must have the same match condition in order to be concatenated,
        otherwise a :py:class:`ValueError` will be raised.
        """
        if (
            left.match_condition == right.match_condition
            and left.match_condition is not None
        ):
            return cls(
                " ".join([left.name, right.name]),
                left.match_skips + right.match_skips,
                left.match_condition,
            )
        else:
            raise ValueError(
                f"Cannot add (mode is None or mode mismatch: "
                f"{left.match_condition=} != {right.match_condition=}): {left=} + {right=}"
            )
