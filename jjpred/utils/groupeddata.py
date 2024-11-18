"""Utilities for handling data that is grouped either by category or channel."""

from collections.abc import ItemsView
from textwrap import TextWrapper
from typing import Protocol, runtime_checkable
from jjpred.channel import Channel
from jjpred.sku import Category


@runtime_checkable
class CategoryGroupProtocol(Protocol):
    def category_exists(self, category: Category) -> bool:
        return category in self.all_categories

    @property
    def all_categories(self) -> list[Category]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return TextWrapper(
            break_on_hyphens=False, replace_whitespace=False
        ).fill(super().__repr__())


class CategoryGroups[T: CategoryGroupProtocol]:
    data: list[T] = []

    def __init__(self, groups: list[T] = []) -> None:
        self.data = groups

    def __repr__(self) -> str:
        # return self.data.__repr__()
        result = []
        result.append("[")
        for x in self.data:
            result.append(
                TextWrapper(
                    break_on_hyphens=False, replace_whitespace=False
                ).fill(str(x.__repr__()))
            )
        result.append("]")
        return "\n".join(result)

    def find_category(self, category: Category) -> int:
        for ix, group in enumerate(self.data):
            if category in group.all_categories:
                return ix
        raise ValueError(f"No groups ({len(self.data)=}) contain {category}.")

    def category_groups(self) -> list[T]:
        return self.data

    def get_category_group(self, category: Category) -> T:
        return self.data[self.find_category(category)]

    def try_get_item(self, category: Category) -> T | None:
        try:
            return self.data[self.find_category(category)]
        except ValueError:
            return None


class ChannelCategoryData[U: CategoryGroups, T: CategoryGroupProtocol]:
    data: dict[Channel, U] = {}

    def __repr__(self) -> str:
        return self.data.__repr__()

    def __init_subclass__(cls) -> None:
        cls.data = {}

    def items(self) -> ItemsView[Channel, U]:
        return self.data.items()

    def channel_exists(self, channel: str | Channel) -> bool:
        ch = Channel.parse(channel)
        return ch in self.data.keys()

    def set_category_group_for_channel(
        self, channel: str | Channel, data: U
    ) -> None:
        ch = Channel.parse(channel)
        if self.channel_exists(ch):
            raise KeyError(f"{str(ch)} already has data.")
        else:
            self.data[ch] = data

    def get_category_groups_for_channel(self, channel: str | Channel) -> U:
        return self.data[Channel.parse(channel)]

    def get_group(self, channel: str | Channel, category: Category) -> T:
        return self.get_category_groups_for_channel(
            channel
        ).get_category_group(category)
