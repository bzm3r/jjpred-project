"""Functions to read form XORO/NetSuite inventory files."""

from __future__ import annotations

from enum import Enum, auto
import polars as pl

from jjpred.analysisdefn import AnalysisDefn
from jjpred.readsupport.nsinventory import (
    get_netsuite_inventory_path,
    read_netsuite_inv,
)
from jjpred.readsupport.xoroinventory import (
    get_xoro_inventory_path,
    read_xoro_inv,
)


class InventoryType(Enum):
    NETSUITE = auto()
    XORO = auto()
    AUTO = auto()


def read_inventory(
    analysis_defn: AnalysisDefn,
    inventory_type: InventoryType,
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
    overwrite: bool = True,
) -> pl.DataFrame:
    if inventory_type == InventoryType.AUTO:
        try:
            get_netsuite_inventory_path(
                analysis_defn.warehouse_inventory_date, verbose=False
            )
            inventory_type = InventoryType.NETSUITE
        except OSError:
            try:
                get_xoro_inventory_path(
                    analysis_defn.warehouse_inventory_date, verbose=False
                )
                inventory_type = InventoryType.XORO
            except OSError:
                raise ValueError(
                    f"Could not find valid NetSuite or XORO inventory path for "
                    f"date {analysis_defn.warehouse_inventory_date}"
                )

    match inventory_type:
        case InventoryType.NETSUITE:
            return read_netsuite_inv(
                analysis_defn,
                read_from_disk=read_from_disk,
                delete_if_exists=delete_if_exists,
                overwrite=overwrite,
            )
        case InventoryType.XORO:
            return read_xoro_inv(
                analysis_defn,
                read_from_disk=read_from_disk,
                delete_if_exists=delete_if_exists,
                overwrite=overwrite,
            )
