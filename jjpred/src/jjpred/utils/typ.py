"""Utilities for dealing with type-related issues in Python."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import Field, dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    Protocol,
    Self,
    runtime_checkable,
)
import polars as pl

type ScalarOrList[T] = T | list[T]
"""It is often convenient to have a function which can accept a scalar, or a
list of scalars."""


def expect_scalar[T](xs: ScalarOrList[T]) -> T:
    """Convert an object that might be scalar or list into a scalar. Raise an
    error if the item is not a scalar."""
    if not isinstance(xs, list):
        return xs
    raise ValueError(f"Expecting scalar, got {xs}.")


def normalize_as_list[T, U](
    xs: ScalarOrList[T] | ScalarOrList[U] | None,
    length: int = 1,
) -> list[T | U]:
    """Convert an object that might be a scalar or list into a list."""
    if isinstance(xs, list):
        return list(x for x in xs)
    elif xs is not None:
        return list(xs for _ in range(length))
    else:
        return list()


def normalize_scalar_or_list_of_sets[T](
    inputs: ScalarOrList[T | set[T]] | None,
    length: int = 1,
) -> list[T]:
    """Convert a scalar or list of sets of scalars into a list."""
    xs: list[T | set[T]] = normalize_as_list(inputs)
    result = []
    for x in xs:
        if isinstance(x, set):
            result += list(x)
        else:
            result.append(x)

    return result


@dataclass
class PolarsLit:
    """Object representing a Polars literal."""

    dtype: pl.DataType = field(default_factory=lambda: pl.Null())
    lit: pl.Expr = field(default_factory=lambda: pl.lit(None))
    value: Any = field(default_factory=lambda: None)

    def __init__(self, value: Any, dtype: pl.DataType | None = None) -> None:
        if dtype is None:
            constructor: Callable[..., pl.DataType] | pl.DataType = (
                pl.DataType.from_python(value.__class__)
            )
            if isinstance(constructor, Callable):
                dtype = constructor()
            else:
                dtype = constructor

            assert isinstance(dtype, pl.DataType)

        self.dtype = dtype
        self.value = value
        self.lit = pl.lit(self.value, dtype=self.dtype)


def as_polars_type[T: pl.DataType | pl.Expr](x: Any, required: type[T]) -> T:
    """Assert that given object is of the required Polars type."""
    if isinstance(x, required):  # required is red-underlined by type checker
        return x
    else:
        raise TypeError(f"{x} is not of type {required}")


def do_nothing[T](x: T) -> T:
    """Takes an input, and returns it. The identity function."""
    return x


def normalize_default_dict[K, T](
    x: T | defaultdict[K, T],
) -> defaultdict[K, T]:
    """Given a scalar convert it into a default dict whose default value is the
    scalar, or if given a default dict, return it unchanged."""
    if isinstance(x, defaultdict):
        return x
    return defaultdict(lambda: x, {})


@runtime_checkable
class Additive(Protocol):
    """:py:class:`Protocol` defining Python objects that can be combined."""

    @classmethod
    def combine(cls, left: Self, right: Self) -> Self:
        raise NotImplementedError()

    @classmethod
    def __combine__(cls, left: Any, right: Any) -> Self:
        if (left is None or left == 0) and isinstance(right, cls):
            return right
        elif isinstance(left, cls) and (right is None or right == 0):
            return left
        elif isinstance(left, cls) and isinstance(right, cls):
            cls.combine(left, right)

        raise ValueError(f"Cannot add: {left=} + {right=}")

    def __add__(self, other: Any) -> Self:
        return self.__class__.__combine__(self, other)

    def __radd__(self, other: Any) -> Self:
        return self.__class__.__combine__(other, self)

    @classmethod
    def sum(cls, defns: list[Self]) -> Self | None:
        if defns and len(defns) > 0:
            return defns[0] + cls.sum(defns[1:])
        else:
            return None


class RuntimeCheckableDataclass(Protocol):
    """Building block for a checkable dataclass that can assert whether its
    fields are being set with the right values."""

    __dataclass_fields__: dict[str, Field]

    @classmethod
    def fields(cls) -> list[str]:
        return list(cls.__dataclass_fields__.keys())

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dataclass_fields__.keys():
            if isinstance(value, pl.DataFrame):  # type: ignore
                super().__setattr__(name, value)
            else:
                raise TypeError(
                    f"Expected value of type {self.__dataclass_fields__[name].type}, got {type(value)}"
                )
        else:
            raise ValueError(f"{name} is not an expected dataclass field")


def as_type[T](x: Any, required_type: type[T]) -> T:
    """Assert that the given object is of the required type."""
    if isinstance(x, required_type):
        return x
    else:
        if isinstance(x, MappingProxyType):
            try:
                return required_type(**x)
            except TypeError:
                pass
        raise TypeError(
            f"Expected value of type {required_type}, got {type(x)}: {x}"
        )


def normalize_optional[T](x: T | None, default: T) -> T:
    """Converts `T | None` into a `T`, given a default value."""
    if x is not None:
        return x
    return default


def create_assert_result(**kwargs: Any) -> dict[str, Any]:
    """Create a dictionary that can be used to nicely print something on the
    fail-path of a Python assert."""
    return kwargs
