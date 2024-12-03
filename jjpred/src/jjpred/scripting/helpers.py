from __future__ import annotations

from jjpred.analysisdefn import AnalysisDefn
from jjpred.structlike import StructLike
from jjpred.database import DataBase


def load_db(
    analysis_defn: AnalysisDefn,
    read_from_disk=True,
    filters: list[StructLike] | None = None,
) -> DataBase:
    database = DataBase(
        analysis_defn,
        read_from_disk=read_from_disk,
        filters=filters,
    )

    return database
