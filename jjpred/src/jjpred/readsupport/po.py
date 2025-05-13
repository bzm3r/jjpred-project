"""Functions to read PO data from static PO files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import NamedTuple

from jjpred.analysisdefn import AnalysisDefn
from jjpred.channel import Channel
from jjpred.globalpaths import BRIAN_TWK_FOLDER
from jjpred.readsheet import fill_unnamed_cols
import polars as pl
import polars.selectors as cs
from jjpred.readsupport.utils import (
    cast_standard,
)
from jjpred.seasons import POSeason, Season
from jjpred.sku import Sku
from jjpred.structlike import MemberType
from jjpred.utils.fileio import (
    delete_or_read_df,
    disable_fastexcel_dtypes_logger,
    gen_support_info_path,
    read_meta_info,
    write_df,
)
from jjpred.utils.polars import (
    vstack_to_unified,
    find_dupes,
    sanitize_excel_extraction,
)
import fastexcel as fxl


# STATIC_PO_FILE: str = "{year_code}{season_code} PO Plan All CATs - static.xlsx"
STATIC_PO_FILE: str = "static_po_sheets.xlsx"


def get_static_po_path() -> Path:
    """Get the path of the relevant static PO file.

    If the file is not found, raise an error."""
    expected_file_name = STATIC_PO_FILE
    file_path = (
        BRIAN_TWK_FOLDER.joinpath("AnalysisData")
        .joinpath("PO")
        .joinpath(expected_file_name)
    )

    if file_path.exists():
        print(f"{file_path} exists!")
        return file_path

    raise OSError(f"Could not find valid static PO file: {file_path}.")


@dataclass
class POSheet:
    season: Season
    year: int
    name: str

    def __init__(self, season: Season, year_code: int, name: str):
        self.season = season
        self.year = 2000 + year_code
        self.name = name


def process_all_po_per_sku(
    all_sku_info: pl.DataFrame,
    all_po_per_sku: pl.DataFrame,
    filter_out_SMF_25F: bool,
) -> pl.DataFrame:
    if filter_out_SMF_25F:
        all_po_per_sku = (
            all_po_per_sku.join(
                all_sku_info.select("sku", "category"), on=["sku"], how="left"
            )
            .filter(
                pl.col.category.is_null()
                | ~(pl.col.po_season.eq("FW") & pl.col.category.eq("SMF"))
            )
            .drop("category")
        )

    return all_po_per_sku


def read_all_po(
    analysis_defn: AnalysisDefn,
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
    treat_negative_po_as_zero: bool = True,
    treat_negative_po_as_error: bool = False,
    overwrite: bool = True,
    filter_out_SMF_25F: bool = True,
) -> pl.DataFrame:
    """Read all PO data from static PO sheets."""

    all_po_per_sku_path = gen_support_info_path(
        analysis_defn,
        "all_po_per_sku",
        analysis_defn.po_date,
        source_name="static_po",
    )

    all_sku_info = read_meta_info(analysis_defn, "all_sku")

    if read_from_disk or delete_if_exists:
        all_po_per_sku = delete_or_read_df(
            delete_if_exists, all_po_per_sku_path
        )
        # per_cat = delete_or_read_df(delete_if_exists, per_cat_path)

        if all_po_per_sku is not None:  # and per_cat is not None:
            return process_all_po_per_sku(
                all_sku_info, all_po_per_sku, filter_out_SMF_25F
            )

    channel_info = read_meta_info(analysis_defn, "channel")

    all_po_per_sku = pl.DataFrame()

    static_po_path = get_static_po_path()

    po_sheet_names = fxl.read_excel(static_po_path).sheet_names
    po_sheets = []
    for sheet_name in po_sheet_names:
        match = re.match(r"(?P<year>\d{2})(?P<season>SS|FW)_PO", sheet_name)
        if match is not None:
            gd = match.groupdict()
            po_sheets.append(
                POSheet(
                    Season.from_str(gd["season"]), int(gd["year"]), sheet_name
                )
            )

    for po_sheet in po_sheets:
        po_headers = pl.read_excel(
            static_po_path,
            sheet_name=po_sheet.name,
            read_options={
                "header_row": 0,
                "n_rows": 1,
            },
        )
        rename_map = fill_unnamed_cols(po_headers, replace_whitespace="_")
        use_columns = []

        intermediate_names = []
        final_names = {}
        possible_channels = [
            "amazon.ca",
            "amazon.com",
            "amazon.uk",
            "amazon.co.uk",
            "amazon.de",
            "amazon.eu",
            "janandjul.com",
        ]
        for k, v in rename_map.items():
            v = v.lower()
            for x in [
                "category",
                "sku",
            ] + possible_channels:
                if x in v and not any([y in v for y in ["ratio"]]):
                    if "amazon" in x or "janandjul" in x:
                        final_names[v] = v.split(" ")[-1]
                        intermediate_names.append(v)
                    else:
                        intermediate_names.append(x.lower())
                    use_columns.append(k)

        raw_season_po = pl.read_excel(
            static_po_path,
            sheet_name=po_sheet.name,
            read_options={
                "header_row": 0,
                "skip_rows": 1,
                "use_columns": use_columns,
            },
        )

        season_po = sanitize_excel_extraction(
            raw_season_po.rename(
                {
                    k: v
                    for k, v in zip(
                        raw_season_po.columns, intermediate_names, strict=True
                    )
                }
            )
        ).with_columns(
            po_season=pl.lit(po_sheet.season.name),
            po_year=pl.lit(po_sheet.year),
        )
        del raw_season_po

        expected_channels = [
            "amazon.com",
            "amazon.ca",
            "amazon.uk",
            "amazon.co.uk",
            "amazon.de",
            "janandjul.com",
        ]

        channel_columns = []
        for x in season_po.columns:
            split_parts = x.split(" ")

            if len(split_parts) > 1:
                try:
                    int_part = int(split_parts[-1])
                except ValueError:
                    int_part = None

                if int_part is not None:
                    x = " ".join(split_parts[:-1])

            ch = Channel.try_from_str(x)
            if ch is not None and ch not in channel_columns:
                channel_columns.append(ch)

        assert all(
            [Channel.parse(x) in channel_columns for x in expected_channels]
        ), [
            x
            for x in expected_channels
            if Channel.parse(x) not in channel_columns
        ]

        for channel in expected_channels:
            ch = Channel.parse(channel)
            this_columns = cs.expand_selector(
                season_po,
                cs.contains(
                    "category", "sku", "po_season", "po_year", channel
                ),
            )

            if len(this_columns) <= 3:
                # we were not able to pick up any channel specific columns
                continue

            sku_info = (
                season_po.select(this_columns)
                .drop("category")
                .rename(
                    {k: v for k, v in final_names.items() if k in this_columns}
                )
                .unpivot(index=["sku", "po_season", "po_year"])
                .rename({"variable": "month", "value": "sales"})
                .cast({"month": pl.Int8()})
                .filter(~pl.col("sku").is_null())
                .with_columns(**ch.to_columns())
            )

            all_po_per_sku = vstack_to_unified(
                all_po_per_sku,
                sku_info,
            )

    all_po_per_sku = cast_standard(
        [all_sku_info, channel_info],
        all_po_per_sku.cast({"po_season": POSeason.polars_type()}),
        strict=False,
    ).drop_nulls()

    if treat_negative_po_as_zero and not treat_negative_po_as_error:
        all_po_per_sku = all_po_per_sku.with_columns(
            sales=pl.when(pl.col.sales.lt(0.0))
            .then(0.0)
            .otherwise(pl.col.sales)
        )
    elif treat_negative_po_as_error:
        sys.displayhook(all_po_per_sku.filter(pl.col.sales.lt(0.0)))
        raise ValueError("Found negative PO values!")

    for x, y in [("HXP-ROS-M", "HXP-ROS-M1"), ("FMR-STB-M", "FMR-STB-M1")]:
        all_po_per_sku = all_po_per_sku.with_columns(
            sku=pl.when(pl.col.sku.eq(x))
            .then(pl.lit(y, dtype=all_po_per_sku["sku"].dtype))
            .otherwise(pl.col.sku)
        )

    all_po_per_sku = process_all_po_per_sku(
        all_sku_info, all_po_per_sku, filter_out_SMF_25F
    )

    write_df(overwrite, all_po_per_sku_path, all_po_per_sku)

    return all_po_per_sku


def relevant_year_per_season_tag(
    relevant_year_per_season: dict[POSeason, int],
) -> str:
    return "_".join(
        [
            f"{year - 2000}{season}"
            for season, year in relevant_year_per_season.items()
        ]
    )


def rename_25SS_gra_to_asg_for_hca0_hcb0(
    active_sku_info: pl.DataFrame, df: pl.DataFrame
) -> pl.DataFrame:
    return cast_standard(
        [active_sku_info],
        df.with_columns(
            print=pl.when(
                pl.col.print.eq("GRA")
                & pl.col.category.is_in(["HCA0", "HCB0"])
                & pl.col.po_year.eq(2025)
            )
            .then(pl.lit("ASG"))
            .otherwise(pl.col.print),
            sku=pl.when(
                pl.col.print.eq("GRA")
                & pl.col.category.is_in(["HCA0", "HCB0"])
                & pl.col.po_year.eq(2025)
            )
            .then(
                pl.concat_str(
                    [
                        pl.col.category.cast(pl.String()),
                        pl.lit("ASG"),
                        pl.col.size.cast(pl.String()),
                    ],
                    separator="-",
                )
            )
            .otherwise(pl.col.sku),
            a_sku=pl.when(
                pl.col.print.eq("GRA")
                & pl.col.category.is_in(["HCA0", "HCB0"])
                & pl.col.po_year.eq(2025)
            )
            .then(
                pl.concat_str(
                    [
                        pl.col.category.cast(pl.String()),
                        pl.lit("ASG"),
                        pl.col.size.cast(pl.String()),
                    ],
                    separator="-",
                )
            )
            .otherwise(pl.col.a_sku),
        ),
    )


def read_po(
    analysis_defn: AnalysisDefn,
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
    overwrite: bool = True,
) -> pl.DataFrame:
    disable_fastexcel_dtypes_logger()

    po_per_sku_path = gen_support_info_path(
        analysis_defn,
        f"po_per_sku_{analysis_defn.current_seasons.tag()}",
        analysis_defn.po_date,
        source_name="static_po",
    )

    if read_from_disk or delete_if_exists:
        po_per_sku = delete_or_read_df(delete_if_exists, po_per_sku_path)

        if po_per_sku is not None:  # and per_cat is not None:
            return rename_25SS_gra_to_asg_for_hca0_hcb0(
                read_meta_info(analysis_defn, "active_sku"), po_per_sku
            )

    all_po_per_sku = read_all_po(
        analysis_defn,
        read_from_disk=read_from_disk,
        delete_if_exists=delete_if_exists,
        overwrite=overwrite,
        filter_out_SMF_25F=True,
    )

    active_sku_info = read_meta_info(analysis_defn, "active_sku")

    relevant_sku_info = active_sku_info.select(
        ["a_sku"]
        + Sku.members(MemberType.META)
        + [
            "is_current_sku",
            "sku_current_year",
            "sku_current_po_season",
        ]
    )

    find_dupes(
        relevant_sku_info,
        ["a_sku"] + Sku.members(MemberType.META),
        raise_error=True,
    )

    all_po_per_sku = (
        all_po_per_sku.join(
            relevant_sku_info,
            on="sku",
            nulls_equal=True,
        )
        .filter(pl.col.is_current_sku)
        .drop("is_current_sku")
        .filter(
            (pl.col.sku_current_year + 2000).eq(pl.col.po_year)
            & pl.col.sku_current_po_season.eq(pl.col.po_season)
        )
    )

    find_dupes(
        all_po_per_sku,
        Sku.members(MemberType.META) + Channel.members() + ["month"],
        raise_error=True,
    )

    all_po_per_sku = rename_25SS_gra_to_asg_for_hca0_hcb0(
        active_sku_info, all_po_per_sku
    )

    write_df(overwrite, po_per_sku_path, all_po_per_sku)

    return all_po_per_sku


class PredictedDemandData(NamedTuple):
    """Predicted demand data read from the main program file."""

    prediction_type: pl.DataFrame
    """Prediction mode (PO, E, NE, CE) used for each SKU."""
    predicted_demand: pl.DataFrame
    """The predicted sales for each SKU."""
