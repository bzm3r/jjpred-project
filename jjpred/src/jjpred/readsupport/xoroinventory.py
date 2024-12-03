"""Functions to read form the ``Item Inventory Snapshot`` XORO inventory file."""

from __future__ import annotations

from pathlib import Path
import polars as pl

from jjpred.analysisdefn import AnalysisDefn, FbaRevDefn
from jjpred.channel import Channel
from jjpred.globalpaths import ANALYSIS_INPUT_FOLDER
from jjpred.readsupport.utils import parse_channels
from jjpred.utils.datetime import Date, DateLike
from jjpred.utils.fileio import (
    delete_or_read_df,
    gen_support_info_path,
    write_df,
)
from jjpred.utils.polars import sanitize_excel_extraction

XORO_INVENTORY_PATH: str = "Item Inventory Snapshot_{date}.xlsx"


def get_xoro_inventory_path(file_date: DateLike, verbose: bool = True) -> Path:
    file_date = Date.from_datelike(file_date)
    inventory_path = ANALYSIS_INPUT_FOLDER.joinpath(
        XORO_INVENTORY_PATH.format(date=file_date.format_as(r"%m%d%Y"))
    )
    if verbose:
        print(f"Checking if {inventory_path} exists...")
    if inventory_path.exists():
        if verbose:
            print("...yes!")
        return inventory_path

    path_shape = XORO_INVENTORY_PATH.format(
        version="{NUMBER}", date=str(file_date)
    )
    raise OSError(
        f"Could not find valid inventory file for {str(file_date)}. "
        f"Should have shape: {path_shape}"
    )


def read_xoro_inv(
    analysis_defn: AnalysisDefn,
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
):
    save_path = gen_support_info_path(
        analysis_defn,
        "inventory",
        analysis_defn.warehouse_inventory_date,
        source_name="xoro",
    )

    if read_from_disk:
        result = delete_or_read_df(delete_if_exists, save_path)
        if result is not None:
            return result

    # active_sku_info = read_meta_info(analysis_defn, "active_sku")
    # channel_info = read_meta_info(analysis_defn, "channel")
    xoro_inv_path = get_xoro_inventory_path(
        analysis_defn.warehouse_inventory_date
    )

    sheet_name = "Table 1"

    use_columns = [
        "Store",
        "Item#",
        "Description",
        "ATS",
        "Standard Unit Price",
        "Active",
        "Item Category",
        "Group",
    ]
    rename_map = {k: k.lower() for k in use_columns} | {
        "Store": "channel",
        "Item#": "sku",
        "ATS": "stock",
        "Standard Unit Price": "price",
        "Item Category": "item_type",
        "Group": "item_group",
    }
    raw_df = (
        sanitize_excel_extraction(
            pl.read_excel(
                xoro_inv_path,
                sheet_name=sheet_name,
                read_options={"use_columns": use_columns},
            )
        )
        .rename(rename_map)
        .filter(pl.col("active").eq("Y"))
        .drop("active")
    )

    xoro_inv_df = parse_channels(raw_df)

    # xoro_inv_df = cast_standard(
    #     [active_sku_info, channel_info],
    #     raw_df.join(unique_channels, on="channel", validate="m:1"),
    #     strict=False,
    # )

    # xoro_inv_df = (
    #     xoro_inv_df.join(
    #         active_sku_info.select(
    #             Sku.members(Priority.META)
    #             + ["pause_plan", "season", "sku_latest_year"]
    #         ),
    #         on="sku",
    #     )
    #     .drop_nulls()
    #     .unique()
    # )

    write_df(True, save_path, xoro_inv_df)

    return xoro_inv_df
