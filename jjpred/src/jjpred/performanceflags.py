from enum import auto
from jjpred.utils.polars import EnumLike


class PerformanceFlag(EnumLike):
    """:py:class:`EnumLike` representing different performance flags that may be
    assigned to a SKU."""

    DISABLED = auto()
    """Over/under-performer logic is disabled."""
    NORMAL = auto()
    """SKU is not over/under-performing."""
    OVER = auto()
    """SKU is doing better than PO prediction estimates."""
    UNDER = auto()
    """SKU is doing worse than PO predictio estimates."""
