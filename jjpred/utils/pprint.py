"""Failed experiments in pretty printing Python objects."""

from pprint import pprint
from typing import Any, Protocol, runtime_checkable


def indent_print(object: str, indent: int = 0, width: int = 80) -> None:
    print(f"{"\t" * indent}{object}".ljust(width))


@runtime_checkable
class PrettyPrint(Protocol):
    def __pprint_items__(self) -> dict[str, Any]:
        raise NotImplementedError()

    def __pprint__(self, indent: int = 0, width: int = 80) -> None:
        for k, item in self.__pprint_items__().items():
            indent_print(k, indent=indent)
            if isinstance(item, PrettyPrint):
                item.pprint(indent=indent + 1)
            else:
                pprint(item, indent=indent, compact=True, width=80)

    def pprint(self, indent: int = 0, width: int = 80):
        indent_print(
            self.__class__.__qualname__ + "(", indent=indent, width=80
        )
        self.__pprint__(indent=indent + 1, width=80)
        indent_print(")", indent=indent, width=80)
