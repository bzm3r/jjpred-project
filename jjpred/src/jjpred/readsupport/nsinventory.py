"""Functions to read form the ``Items_S`` NetSuite inventory file."""

from __future__ import annotations

from pathlib import Path
import polars as pl

from jjpred.analysisdefn import AnalysisDefn
from jjpred.channel import KNOWN_CHANNEL_MATCHERS
from jjpred.globalpaths import ANALYSIS_INPUT_FOLDER
from jjpred.readsupport.utils import parse_channels
from jjpred.utils.datetime import Date, DateLike
from jjpred.utils.fileio import (
    delete_or_read_df,
    gen_support_info_path,
    write_df,
)
from jjpred.utils.polars import sanitize_excel_extraction

NS_SURREY_INV_PATH: str = "Items_S_{date}.csv"
NS_INV_PATH: str = "Items_All_{date}.csv"


def get_ns_surrey_inventory_path(
    file_date: DateLike, verbose: bool = True
) -> Path:
    file_date = Date.from_datelike(file_date)
    file_date_str = file_date.format_as(r"%Y%m%d")
    inventory_path = ANALYSIS_INPUT_FOLDER.joinpath(
        NS_SURREY_INV_PATH.format(date=file_date_str)
    )
    if verbose:
        print(f"Checking if {inventory_path} exists...")
    if inventory_path.exists():
        if verbose:
            print("...yes!")
        return inventory_path

    path_shape = NS_SURREY_INV_PATH.format(date=file_date_str)
    raise OSError(
        f"Could not find valid inventory file for {str(file_date)}. "
        f"Should have shape: {path_shape}"
    )


def get_ns_inventory_path(file_date: DateLike, verbose: bool = True) -> Path:
    file_date = Date.from_datelike(file_date)
    file_date_str = file_date.format_as(r"%Y%m%d")
    inventory_path = ANALYSIS_INPUT_FOLDER.joinpath(
        NS_INV_PATH.format(date=file_date_str)
    )
    if verbose:
        print(f"Checking if {inventory_path} exists...")
    if inventory_path.exists():
        if verbose:
            print("...yes!")
        return inventory_path

    path_shape = NS_INV_PATH.format(date=file_date_str)
    raise OSError(
        f"Could not find valid inventory file for {str(file_date)}. "
        f"Should have shape: {path_shape}"
    )


def read_netsuite_surrey_inv(
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

    inventory_path = get_ns_surrey_inventory_path(
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

    ns_inv_df = parse_channels(raw_df)

    write_df(overwrite, save_path, ns_inv_df)

    return ns_inv_df


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

    inventory_path = get_ns_inventory_path(
        analysis_defn.warehouse_inventory_date
    )

    use_columns = {
        "Name": pl.String(),
        "Inventory Warehouse": pl.String(),
        "Warehouse Available": pl.Int64(),
        "Item Parent Category": pl.String(),
        "Base Price": pl.String(),
        "Item Category SKU": pl.String(),
    }
    rename_map = {k: k.lower() for k in use_columns.keys()} | {
        "Name": "sku",
        "Inventory Warehouse": "channel",
        "Warehouse Available": "stock",
        "Item Parent Category": "item_category",
        "Base Price": "price",
        "Item Category SKU": "category",
    }
    raw_df = sanitize_excel_extraction(
        pl.read_csv(
            inventory_path,
            has_header=True,
            columns=list(use_columns.keys()),
            schema_overrides=use_columns,
        )
        .filter(~pl.col("Base Price").str.ends_with("%"))
        .with_columns(pl.col("Base Price").cast(pl.Float64()))
    ).rename(rename_map)

    ns_inv_df = parse_channels(
        raw_df.filter(
            pl.col.channel.str.to_lowercase().is_in(
                [x.lower() for x in KNOWN_CHANNEL_MATCHERS.keys()]
            )
        )
    )

    write_df(overwrite, save_path, ns_inv_df)

    return ns_inv_df
