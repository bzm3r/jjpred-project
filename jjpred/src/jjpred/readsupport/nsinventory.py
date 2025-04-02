"""Functions to read form the ``Items_S`` NetSuite inventory file."""

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

NETSUITE_INVENTORY_PATH: str = "Items_S_{date}.csv"


def get_netsuite_inventory_path(
    file_date: DateLike, verbose: bool = True
) -> Path:
    file_date = Date.from_datelike(file_date)
    file_date_str = file_date.format_as(r"%Y%m%d")
    inventory_path = ANALYSIS_INPUT_FOLDER.joinpath(
        NETSUITE_INVENTORY_PATH.format(date=file_date_str)
    )
    if verbose:
        print(f"Checking if {inventory_path} exists...")
    if inventory_path.exists():
        if verbose:
            print("...yes!")
        return inventory_path

    path_shape = NETSUITE_INVENTORY_PATH.format(date=str(file_date))
    raise OSError(
        f"Could not find valid inventory file for {str(file_date)}. "
        f"Should have shape: {path_shape}"
    )


def read_netsuite_inv(
    analysis_defn: AnalysisDefn,
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
    overwrite: bool = True,
):
    save_path = gen_support_info_path(
        analysis_defn,
        "inventory",
        analysis_defn.warehouse_inventory_date,
        source_name="netsuite",
    )

    if read_from_disk:
        result = delete_or_read_df(delete_if_exists, save_path)
        if result is not None:
            return result

    # active_sku_info = read_meta_info(analysis_defn, "active_sku")
    # channel_info = read_meta_info(analysis_defn, "channel")
    inventory_path = get_netsuite_inventory_path(
        analysis_defn.warehouse_inventory_date
    )

    use_columns = [
        "Name",
        "Warehouse Available",
        "Item Parent Category",
        "Base Price",
        "Item Category SKU",
    ]
    rename_map = {k: k.lower() for k in use_columns} | {
        "Name": "sku",
        "Warehouse Available": "stock",
        "Item Parent Category": "item_category",
        "Base Price": "price",
        "Item Category SKU": "category",
    }
    raw_df = (
        sanitize_excel_extraction(
            pl.read_csv(
                inventory_path,
                has_header=True,
                columns=use_columns,
            )
        )
        .rename(rename_map)
        .with_columns(channel=pl.lit("Warehouse CA"))
    )
    # unique_channels = (
    #     pl.DataFrame(raw_df["channel"].unique())
    #     .with_columns(
    #         struct_channel=pl.col("channel").map_elements(
    #             Channel.map_polars, return_dtype=Channel.polars_type_struct()
    #         )
    #     )
    #     .unnest("struct_channel")
    # )

    ns_inv_df = parse_channels(raw_df)

    write_df(overwrite, save_path, ns_inv_df)

    return ns_inv_df
