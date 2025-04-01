"""Functions to read PO data from static PO files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Literal, NamedTuple

from jjpred.analysisdefn import AnalysisDefn, RefillDefn
from jjpred.channel import Channel
from jjpred.datagroups import ALL_SKU_AND_CHANNEL_IDS
from jjpred.globalpaths import BRIAN_TWK_FOLDER
from jjpred.readsheet import fill_unnamed_cols
import polars as pl
import polars.selectors as cs
from jjpred.readsupport.utils import (
    cast_standard,
)
from jjpred.seasons import Season
from jjpred.sku import Sku
from jjpred.structlike import MemberType
from jjpred.utils.datetime import (
    Date,
    DateLike,
)
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


def get_po_season(
    dispatch_date: Date,
) -> (
    tuple[Literal[Season.FW], Literal[Season.SS]]
    | tuple[Literal[Season.SS], Literal[Season.FW]]
):
    if 3 <= dispatch_date.month and dispatch_date.month <= 9:
        return Season.SS, Season.FW
    else:
        return Season.FW, Season.SS


@dataclass
class POSheet:
    season: Season
    year: int
    name: str

    def __init__(self, season: Season, year_code: int, name: str):
        self.season = season
        self.year = 2000 + year_code
        self.name = name


def read_all_po(
    analysis_defn: AnalysisDefn,
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
    treat_negative_po_as_zero: bool = True,
    treat_negative_po_as_error: bool = False,
) -> pl.DataFrame:
    """Read all PO data from static PO sheets."""

    all_po_per_sku_path = gen_support_info_path(
        analysis_defn,
        "all_po_per_sku",
        analysis_defn.po_date,
        source_name="static_po",
    )

    if read_from_disk or delete_if_exists:
        all_po_per_sku = delete_or_read_df(
            delete_if_exists, all_po_per_sku_path
        )
        # per_cat = delete_or_read_df(delete_if_exists, per_cat_path)

        if all_po_per_sku is not None:  # and per_cat is not None:
            return all_po_per_sku

    all_sku_info = read_meta_info(analysis_defn, "all_sku")
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
        for k, v in rename_map.items():
            v = v.lower()
            for x in [
                "category",
                "sku",
                "amazon.com",
                "amazon.ca",
                "amazon.uk",
                "amazon.co.uk",
                "amazon.de",
                "janandjul.com",
            ]:
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

        for channel in [
            "amazon.com",
            "amazon.ca",
            "amazon.uk",
            "amazon.co.uk",
            "amazon.de",
            "janandjul.com",
        ]:
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

    category_seasons = all_sku_info.select("category", "season").unique()
    find_dupes(category_seasons, ["category", "season"], raise_error=True)

    all_po_per_sku = cast_standard(
        [all_sku_info, channel_info],
        all_po_per_sku,
        strict=False,
        use_dtype_of={"po_season": "season"},
    ).drop_nulls()

    all_po_per_sku = all_po_per_sku.with_columns(
        latest_po_year=pl.col.po_year.max().over("sku", "po_season")
    )

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

    write_df(True, all_po_per_sku_path, all_po_per_sku)

    return all_po_per_sku


def relevant_year_per_season_tag(
    relevant_year_per_season: dict[Literal[Season.FW, Season.SS], int],
) -> str:
    return "_".join(
        [
            f"{year - 2000}{season}"
            for season, year in relevant_year_per_season.items()
        ]
    )


def read_po(
    analysis_defn_and_dispatch_date: RefillDefn
    | tuple[AnalysisDefn, DateLike],
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
) -> pl.DataFrame:
    disable_fastexcel_dtypes_logger()

    if isinstance(analysis_defn_and_dispatch_date, RefillDefn):
        analysis_defn = analysis_defn_and_dispatch_date
        dispatch_date = analysis_defn_and_dispatch_date.dispatch_date
    else:
        analysis_defn = analysis_defn_and_dispatch_date[0]
        dispatch_date = Date.from_datelike(analysis_defn_and_dispatch_date[1])

    dispatch_season, other_season = get_po_season(dispatch_date)

    relevant_year_per_season = {}
    for season in [Season.FW, Season.SS]:
        if season == dispatch_season:
            relevant_year_per_season[season] = dispatch_date.year
        else:
            relevant_year_per_season[season] = dispatch_date.year - 1

    po_per_sku_path = gen_support_info_path(
        analysis_defn,
        f"po_per_sku_{relevant_year_per_season_tag(relevant_year_per_season)}",
        analysis_defn.po_date,
        source_name="static_po",
    )

    if read_from_disk or delete_if_exists:
        po_per_sku = delete_or_read_df(delete_if_exists, po_per_sku_path)
        # per_cat = delete_or_read_df(delete_if_exists, per_cat_path)

        if po_per_sku is not None:  # and per_cat is not None:
            return po_per_sku

    all_po_per_sku = read_all_po(
        analysis_defn,
        read_from_disk=read_from_disk,
        delete_if_exists=delete_if_exists,
    )

    active_sku_info = read_meta_info(analysis_defn, "active_sku")

    all_po_per_sku = all_po_per_sku.join(
        active_sku_info.select(
            ["a_sku"]
            + Sku.members(MemberType.META)
            + [
                "pause_plan",
                "season",
                "sku_year_history",
                "category_year_history",
                "sku_latest_year",
            ]
        ),
        on="sku",
        nulls_equal=True,
    )

    # relevant_po_per_sku = all_po_per_sku.filter(
    #     pl.col.po_year == pl.col.latest_po_year
    # )

    if relevant_year_per_season is None:
        relevant_po_per_sku = all_po_per_sku.filter(
            pl.col.po_year == pl.col.latest_po_year
        )
    else:
        parts = []
        for season in [Season.FW, Season.SS]:
            year = relevant_year_per_season.get(season)
            if year is not None:
                parts.append(
                    all_po_per_sku.filter(
                        pl.col.po_season.eq(str(season)),
                        pl.col.po_year.eq(year),
                    )
                )
            else:
                parts.append(
                    all_po_per_sku.filter(
                        pl.col.po_season.eq(str(season)),
                        pl.col.po_year == pl.col.latest_po_year,
                    )
                )
        relevant_po_per_sku = pl.concat(parts)

    dispatch_season_info = relevant_po_per_sku.filter(
        pl.col.po_season.eq(dispatch_season.name)
    )
    other_season_info = relevant_po_per_sku.filter(
        pl.col.po_season.eq(other_season.name)
    )

    find_dupes(
        dispatch_season_info,
        ALL_SKU_AND_CHANNEL_IDS + ["month"],
        raise_error=True,
    )
    find_dupes(
        other_season_info,
        ALL_SKU_AND_CHANNEL_IDS + ["month"],
        raise_error=True,
    )

    in_dispatch_season_info = dispatch_season_info["sku"].unique()
    other_season_info = other_season_info.with_columns(
        in_dispatch_season_info=pl.col.sku.is_in(in_dispatch_season_info)
    ).filter(~pl.col.in_dispatch_season_info)

    po_per_sku = dispatch_season_info.vstack(
        other_season_info.select(dispatch_season_info.columns)
    )

    find_dupes(
        po_per_sku,
        Sku.members(MemberType.META) + Channel.members() + ["month"],
        raise_error=True,
    )

    write_df(True, po_per_sku_path, po_per_sku)

    return po_per_sku


class PredictedDemandData(NamedTuple):
    """Predicted demand data read from the main program file."""

    prediction_type: pl.DataFrame
    """Prediction mode (PO, E, NE, CE) used for each SKU."""
    predicted_demand: pl.DataFrame
    """The predicted sales for each SKU."""
