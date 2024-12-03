"""Enumerations of the different kind of prediction modes we can expect to read
from input files."""

from __future__ import annotations

from enum import auto, unique

from jjpred.utils.polars import EnumLike


@unique
class InputPredictionType(EnumLike):
    PO = auto()
    CE = auto()
    E = auto()

    def __hash__(self) -> int:
        return self.value.__hash__()


@unique
class PredictionType(EnumLike):
    PO = auto()
    CE = auto()
    NE = auto()
    E = auto()

    def __hash__(self) -> int:
        return self.value.__hash__()
