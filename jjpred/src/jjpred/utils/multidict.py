"""Utilities for compressing a dictionary into a tuple-valued dictionary.

For example, suppose we have a dictionary: ``{"a": 0, "b": 0, "c": 1}``, then we
can interconvert it between ``{("a", "b"): 0, ("c",): 1}`` ("multi-dict")."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self


@dataclass
class MultiDict[T, U]:
    data: dict[tuple[T, ...], U]

    def get(self, key: T) -> U:
        for keys, item in self.data.items():
            if key in keys:
                return item

        raise ValueError(
            f"{key} not in any available keys: {list(self.data.keys())}"
        )

    def __or__(self, other: Any) -> Self:
        if isinstance(other, self.__class__):
            self_dict = self.as_dict()
            other_dict = other.as_dict()
            self_keys = set(self_dict.keys())

            common_keys = self_keys.intersection(other_dict.keys())
            for k in common_keys:
                assert self_dict[k] == other_dict[k], (
                    self_dict[k],
                    other_dict[k],
                )

            return self.__class__.from_dict(self_dict | other_dict)
        else:
            raise ValueError(f"Cannot combine: {self=} | {other=}")

    def as_dict(
        self, default_factory: Callable[..., U] | None = None
    ) -> dict[T, U] | defaultdict[T, U]:
        if default_factory is not None:
            result = defaultdict(default_factory)
        else:
            result = {}

        for key, item in self.data.items():
            if isinstance(key, tuple):
                for k in key:
                    result[k] = item
            else:
                result[key] = item

        return result

    @classmethod
    def from_dict(cls, d: dict[T, U]) -> Self:
        result_keys: list[list[T]] = []
        result_items: list[U] = []
        for key, item in d.items():
            try:
                idx = result_items.index(item)
                result_keys[idx].append(key)
            except ValueError:
                result_keys.append([key])
                result_items.append(item)
        result = {}
        for keys, item in zip(result_keys, result_items):
            result[tuple(keys)] = item
        return cls(result)
