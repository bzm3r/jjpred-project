"""Functions to read from the ``FBA Inventory Opimization Recall and Replenish``
("main program") file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from time import strptime
from typing import Literal, NamedTuple

from jjpred.analysisdefn import AnalysisDefn, RefillDefn
from jjpred.channel import Channel
from jjpred.datagroups import ALL_SKU_AND_CHANNEL_IDS
from jjpred.globalpaths import ANALYSIS_INPUT_FOLDER
from jjpred.readsheet import fill_unnamed_cols
import polars as pl
import polars.selectors as cs
from jjpred.readsupport.utils import (
    NA_FBA_SHEET,
    cast_standard,
)
from jjpred.seasons import Season
from jjpred.sku import Sku
from jjpred.structlike import MemberType
from jjpred.utils.datetime import (
    Date,
    DateLike,
    first_day_next_month,
)
from jjpred.utils.fileio import (
    delete_or_read_df,
    gen_support_info_path,
    read_meta_info,
    write_df,
)
from jjpred.utils.polars import (
    vstack_to_unified,
    find_dupes,
    sanitize_excel_extraction,
)
from jjpred.utils.typ import normalize_as_list, as_polars_type
import fastexcel as fxl


MAIN_PROGRAM_FILE: str = (
    "FBA Inventory Opimization Recall and Replenish v{version} ({date}).xlsm"
)


def get_mainprogram_path(
    file_date: DateLike, max_tries: int = 30, start: int = 15
) -> Path:
    """Get the path of the main program file (``FBA Inventory Opimization Recall
    and Replenish``) with the latest version and matching date.

    If the file is not found, raise an error."""

    assert start > 0

    file_date = Date.from_datelike(file_date)
    for version in reversed(range(start, start + max_tries)):
        calculation_path = ANALYSIS_INPUT_FOLDER.joinpath(
            MAIN_PROGRAM_FILE.format(version=version, date=str(file_date))
        )
        if calculation_path.exists():
            print(f"{calculation_path} exists!")
            return calculation_path

    path_shape = MAIN_PROGRAM_FILE.format(
        version="{NUMBER}", date=str(file_date)
    )
    raise OSError(
        f"Could not find valid calculation file for {str(file_date)}. "
        f"Should have shape: {path_shape}"
    )


def get_po_season(
    dispatch_date: Date,
) -> (
    tuple[Literal[Season.FW], Literal[Season.SS]]
    | tuple[Literal[Season.SS], Literal[Season.FW]]
):
    if dispatch_date.month <= 3 and dispatch_date.month <= 9:
        return Season.SS, Season.FW
    else:
        return Season.FW, Season.SS


@dataclass
class CandidatePOSheet:
    season: Season
    year: int
    name: str


def read_po(
    analysis_defn_and_dispatch_date: RefillDefn
    | tuple[AnalysisDefn, DateLike],
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
    overwrite: bool = True,
) -> pl.DataFrame:
    """Read PO data from the main program ``FBA Inventory Opimization Recall
    and Replenish`` file."""

    if isinstance(analysis_defn_and_dispatch_date, RefillDefn):
        analysis_defn = analysis_defn_and_dispatch_date
        dispatch_date = analysis_defn_and_dispatch_date.dispatch_date
    else:
        analysis_defn = analysis_defn_and_dispatch_date[0]
        dispatch_date = Date.from_datelike(analysis_defn_and_dispatch_date[1])

    per_sku_path = gen_support_info_path(
        analysis_defn,
        "per_sku_po",
        dispatch_date,
        source_name="mainprogram",
    )
    # per_cat_path = gen_support_info_path(
    #     analysis_defn, "per_cat_po", mainprogram_date
    # )
    if read_from_disk or delete_if_exists:
        per_sku = delete_or_read_df(delete_if_exists, per_sku_path)
        # per_cat = delete_or_read_df(delete_if_exists, per_cat_path)

        if per_sku is not None:  # and per_cat is not None:
            return per_sku

    if isinstance(analysis_defn_and_dispatch_date, RefillDefn):
        active_sku_info = read_meta_info(
            analysis_defn_and_dispatch_date, "active_sku"
        )
        channel_info = read_meta_info(
            analysis_defn_and_dispatch_date, "channel"
        )
        mainprogram_path = get_mainprogram_path(
            analysis_defn_and_dispatch_date.get_mainprogram_date()
        )
    else:
        active_sku_info = read_meta_info(
            analysis_defn_and_dispatch_date[0], "active_sku"
        )
        channel_info = read_meta_info(
            analysis_defn_and_dispatch_date[0], "channel"
        )
        mainprogram_path = get_mainprogram_path(
            analysis_defn_and_dispatch_date[1]
        )

    per_cat = pl.DataFrame()
    per_sku = pl.DataFrame()
    seasons = [Season.SS, Season.FW]

    main_program_sheet_names = fxl.read_excel(mainprogram_path).sheet_names
    candidate_sheets = []
    for sheet_name in main_program_sheet_names:
        match = re.match(r"(?P<year>\d{2})(?P<season>SS|FW)_PO", sheet_name)
        if match is not None:
            gd = match.groupdict()
            candidate_sheets.append(
                CandidatePOSheet(
                    Season.from_str(gd["season"]), int(gd["year"]), sheet_name
                )
            )

    season_sheets: dict[Season, CandidatePOSheet] = {}
    for season in seasons:
        for sheet in candidate_sheets:
            if sheet.season == season:
                existing_sheet = season_sheets.get(season)
                if existing_sheet is None or existing_sheet.year < sheet.year:
                    season_sheets[season] = sheet

    for existing_sheet in season_sheets.values():
        po_headers = pl.read_excel(
            mainprogram_path,
            sheet_name=existing_sheet.name,
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
            ]:
                if x in v and not any([y in v for y in ["ratio"]]):
                    if "amazon" in x:
                        final_names[v] = v.split(" ")[-1]
                        intermediate_names.append(v)
                    else:
                        intermediate_names.append(x.lower())
                    use_columns.append(k)

        raw_season_po = pl.read_excel(
            mainprogram_path,
            sheet_name=existing_sheet.name,
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
        ).with_columns(po_season=pl.lit(existing_sheet.season.name))
        del raw_season_po

        for channel in [
            "amazon.com",
            "amazon.ca",
            "amazon.uk",
            "amazon.co.uk",
            "amazon.de",
        ]:
            ch = Channel.parse(channel)
            this_columns = cs.expand_selector(
                season_po, cs.contains("category", "sku", "po_season", channel)
            )

            if len(this_columns) <= 3:
                # we were not able to pick up any channel specific columns
                continue

            cat_info = (
                season_po.select(this_columns)
                .rename(
                    {k: v for k, v in final_names.items() if k in this_columns}
                )
                .unpivot(index=["category", "sku", "po_season"])
                .rename({"variable": "month", "value": "sales"})
                .cast({"month": pl.Int8()})
                .filter(pl.col("sku").is_null())
                .with_columns(**ch.to_columns())
                .drop("sku")
            )

            sku_info = (
                season_po.select(this_columns)
                .drop("category")
                .rename(
                    {k: v for k, v in final_names.items() if k in this_columns}
                )
                .unpivot(index=["sku", "po_season"])
                .rename({"variable": "month", "value": "sales"})
                .cast({"month": pl.Int8()})
                .filter(~pl.col("sku").is_null())
                .with_columns(**ch.to_columns())
            )

            per_cat = vstack_to_unified(
                per_cat,
                cat_info,
            )
            per_sku = vstack_to_unified(
                per_sku,
                sku_info,
            )

    category_seasons = active_sku_info.select("category", "season").unique()
    find_dupes(category_seasons, ["category"], raise_error=True)
    per_cat = (
        cast_standard(
            [active_sku_info, channel_info],
            per_cat,
            strict=False,
            use_dtype_of={"po_season": "season"},
        )
        .drop_nulls()
        .join(
            category_seasons,
            on="category",
            how="left",
            validate="m:1",
        )
        .filter(
            pl.col("po_season").eq(pl.col("season"))
            | pl.col("season").eq("AS")
        )
        .drop("po_season")
        .group_by(cs.exclude("sales"))
        .agg(pl.col("sales").max())
        .with_columns(pl.col("sales"))
    )
    per_sku = cast_standard(
        [active_sku_info, channel_info],
        per_sku,
        strict=False,
        use_dtype_of={"po_season": "season"},
    ).drop_nulls()

    # TODO: the logic here is incorrect. Correct logic should be more like:
    # if dispatch season is X (e.g. X=SS) and Y is other season (e.g. Y=FW):
    #     - then check to see the X sheet for items
    #     - for those item not in the X sheet, check the Y sheet
    per_sku = per_sku.join(
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

    dispatch_season, other_season = get_po_season(dispatch_date)

    dispatch_season_info = per_sku.filter(
        pl.col.po_season.eq(dispatch_season.name)
    )
    other_season_info = per_sku.filter(pl.col.po_season.eq(other_season.name))
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

    per_sku = dispatch_season_info.vstack(
        other_season_info.select(dispatch_season_info.columns)
    )

    find_dupes(
        per_cat,
        ["category"] + Channel.members() + ["month"],
        raise_error=True,
    )
    find_dupes(
        per_sku,
        Sku.members(MemberType.META) + Channel.members() + ["month"],
        raise_error=True,
    )

    write_df(overwrite, per_sku_path, per_sku)
    # write_df(True, per_cat_path, per_cat)

    return per_sku


class PredictedDemandData(NamedTuple):
    """Predicted demand data read from the main program file."""

    prediction_type: pl.DataFrame
    """Prediction mode (PO, E, NE, CE) used for each SKU."""
    predicted_demand: pl.DataFrame
    """The predicted sales for each SKU."""


def read_excel_predictions(
    analysis_defn: RefillDefn,
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
    overwrite: bool = True,
) -> pl.DataFrame:
    """Read predicted sales information from the main program ``FBA Inventory
    Opimization Recall and Replenish`` file."""
    save_path = gen_support_info_path(
        analysis_defn,
        "excel_predictions",
        analysis_defn.get_mainprogram_date(),
        source_name="mainprogram",
    )

    if read_from_disk:
        result = delete_or_read_df(delete_if_exists, save_path)
        if result is not None:
            return result

    active_sku_info = read_meta_info(analysis_defn, "active_sku")
    channel_info = read_meta_info(analysis_defn, "channel")
    mainprogram_path = get_mainprogram_path(
        analysis_defn.get_mainprogram_date()
    )

    sheet_headers = pl.read_excel(
        mainprogram_path,
        sheet_name=NA_FBA_SHEET,
        read_options={
            "header_row": 1,
            "n_rows": 1,
        },
    )

    rename_map = fill_unnamed_cols(sheet_headers, replace_whitespace="_")
    use_columns = []

    intermediate_names = []
    final_names = {}
    for k, v in rename_map.items():
        for x in ["merchant", "status", "amazon.com", "amazon.ca"]:
            if x in v.lower() and not any([y in v.lower() for y in ["fba"]]):
                if "amazon" in x:
                    splits = v.split(" ")
                    if len(splits) == 2:
                        try:
                            month_part = int(splits[-1])
                            if v not in intermediate_names:
                                intermediate_name = v
                            else:
                                month_part = month_part + 12
                                intermediate_name = (
                                    splits[0] + " " + str(month_part)
                                )
                            final_names[intermediate_name] = str(month_part)
                            intermediate_names.append(intermediate_name)
                            use_columns.append(k)
                        except ValueError:
                            pass
                else:
                    if x == "status":
                        intermediate_names.append("category")
                    elif x == "merchant":
                        intermediate_names.append("sku")
                    use_columns.append(k)

    raw_df = pl.read_excel(
        mainprogram_path,
        sheet_name=NA_FBA_SHEET,
        read_options={
            "header_row": 1,
            "use_columns": use_columns,
        },
    )

    df = sanitize_excel_extraction(
        raw_df.rename(
            {
                k: v
                for k, v in zip(
                    raw_df.columns, intermediate_names, strict=True
                )
            }
        )
    )
    del raw_df

    prediction_type = pl.DataFrame()
    predicted_demand = pl.DataFrame()
    for channel in ["Amazon.com", "Amazon.ca"]:
        ch = Channel.parse(channel)

        this_columns = cs.expand_selector(
            df, cs.contains("category", "sku", channel)
        )

        this_df = df.select(this_columns).rename(
            {k: v for k, v in final_names.items() if k in this_columns}
        )

        pd_type_info = (
            this_df.drop("sku")
            .with_columns(pl.col("category").forward_fill(1))
            .filter(pl.col("category").is_not_null())
            .filter(
                pl.col("category").is_in(
                    as_polars_type(
                        active_sku_info["category"].dtype, pl.Enum
                    ).categories
                )
            )
            .with_row_index()
            .filter(
                pl.col("index").mod(2) == 1,
            )
            .drop("index")
            .unpivot(index="category")
            .rename({"variable": "month", "value": "prediction_type"})
            .cast({"month": pl.Int8()})
            .with_columns(**ch.to_columns())
        )
        pd_type_info = pd_type_info.cast(
            {
                "prediction_type": pl.Enum(
                    pd_type_info["prediction_type"].unique().sort()
                )
            }
        )

        pd_info = (
            this_df.drop("category")
            .filter(pl.col("sku").is_not_null())
            .filter(
                pl.col("sku").is_in(
                    as_polars_type(
                        active_sku_info["sku"].dtype, pl.Enum
                    ).categories
                )
            )
            .unpivot(index="sku")
            .rename({"variable": "month", "value": "predicted_demand"})
            .cast({"month": pl.Int8()})
            .cast({"predicted_demand": pl.Int64()}, strict=False)
            .with_columns(**{k: pl.lit(v) for k, v in ch.as_dict().items()})
        )

        prediction_type = vstack_to_unified(
            prediction_type,
            cast_standard([active_sku_info, channel_info], pd_type_info),
        )
        predicted_demand = vstack_to_unified(
            predicted_demand,
            cast_standard([active_sku_info, channel_info], pd_info),
        )

    # sys.displayhook(
    #     find_dupes(active_sku_info.select(Sku.members(Priority.META)), ["sku"])
    # )
    # sys.displayhook(find_dupes(predicted_demand, ["sku"]))
    predicted_demand = predicted_demand.join(
        active_sku_info.select(Sku.members(MemberType.META)),
        on="sku",
        # each SKU on the LHS is repeated many times (for example, per month)
        validate="m:1",
        nulls_equal=True,
    )
    result = predicted_demand.join(
        prediction_type,
        on=["category", "month"] + Channel.members(),
        # many SKUs on LHS could have the same category
        validate="m:1",
        nulls_equal=True,
    )

    write_df(overwrite, save_path, result)

    return result


def parse_current_period_defn(x: str) -> str | None:
    """Parse a current period definition in year-month (e.g. ``2024,2``) style
    into a ``year-month-day`` style string (e.g. ``2024-02-01``)."""
    try:
        result = strptime(x, "%Y,%m")
        return f"{result.tm_year}-{result.tm_mon}-{result.tm_mday}"
    except ValueError:
        return None


def read_current_period_defn(
    analysis_defn: RefillDefn,
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
    overwrite: bool = True,
):
    """Read current period definitions for each category from the main program
    file (``FBA Inventory Opimization Recall and Replenish``)."""
    # ([\w_]+\()
    # [\s\n]*("[\w_]+"),[\s\n]*analysis_defn((?:,?)[\s\n\w_]*\))
    save_path = gen_support_info_path(
        analysis_defn,
        "current_period_defn",
        analysis_defn.get_mainprogram_date(),
        source_name="mainprogram",
    )

    if read_from_disk:
        result = delete_or_read_df(delete_if_exists, save_path)
        if result is not None:
            return result

    active_sku_info = read_meta_info(analysis_defn, "active_sku")
    channel_info = read_meta_info(analysis_defn, "channel")
    mainprogram_path = get_mainprogram_path(
        analysis_defn.get_mainprogram_date()
    )

    sheet_headers = pl.read_excel(
        mainprogram_path,
        sheet_name=NA_FBA_SHEET,
        read_options={
            "header_row": 1,
            "n_rows": 0,
        },
    )

    use_columns = []
    intermediate_names = []
    for col in sheet_headers.columns:
        for x in ["status", ["manual", "adjust"]]:
            if all([y in col.lower() for y in normalize_as_list(x)]):
                if x == "status":
                    intermediate_names.append("category")
                else:
                    intermediate_names.append("current_period")
                use_columns.append(col)

    raw_df = sanitize_excel_extraction(
        pl.read_excel(
            mainprogram_path,
            sheet_name=NA_FBA_SHEET,
            read_options={
                "header_row": 1,
                "use_columns": use_columns,
                "skip_rows": 1,
            },
        ).rename(dict(zip(use_columns, intermediate_names, strict=True)))
    )

    current_period_df = (
        raw_df.filter(
            pl.col("category")
            .is_in(
                as_polars_type(
                    active_sku_info["category"].dtype, pl.Enum
                ).categories
            )
            .or_(pl.col("category").is_null())
        )
        .with_columns(pl.col("category").forward_fill(1))
        .with_row_index()
        .filter(pl.col("index").mod(2).eq(1))
        .drop("index")
        .with_columns(current_period=pl.col("current_period").str.split("-"))
        .with_columns(
            raw_start=pl.col("current_period").list.first(),
            raw_end=pl.col("current_period").list.last(),
        )
        .drop("current_period")
        .with_columns(
            pl.col("raw_" + x)
            .map_elements(parse_current_period_defn, return_dtype=pl.String())
            .str.to_date(r"%Y-%m-%d")
            .alias(f"{x}")
            for x in ["start", "end"]
        )
        .with_columns(int_end=pl.col("raw_end").cast(pl.Int8(), strict=False))
        .with_columns(
            end=pl.when(
                pl.col("end")
                .is_null()
                .and_(pl.col("start").is_not_null())
                .and_(pl.col("int_end").is_not_null())
            )
            .then(pl.date(pl.col("start").dt.year(), pl.col("int_end"), 1))
            .otherwise(pl.col("end"))
        )
        .drop("raw_start", "raw_end", "int_end")
        .with_columns(
            pl.col("end").map_elements(
                lambda x: first_day_next_month(x).date, return_dtype=pl.Date()
            )
        )
        .with_columns(
            pl.col("start").fill_null(pl.date(1, 1, 1)),
            pl.col("end").fill_null(pl.date(1, 1, 1)),
        )
    )

    current_period_df = cast_standard(
        [active_sku_info, channel_info], current_period_df
    )

    write_df(overwrite, save_path, current_period_df)

    return current_period_df


def read_qty_box(
    analysis_defn: RefillDefn,
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
    overwrite: bool = True,
):
    """Read qty/box information from the main program file."""
    save_path = gen_support_info_path(
        analysis_defn,
        "qty_box",
        analysis_defn.get_mainprogram_date(),
        source_name="mainprogram",
    )

    if read_from_disk:
        result = delete_or_read_df(delete_if_exists, save_path)
        if result is not None:
            return result

    active_sku_info = read_meta_info(analysis_defn, "active_sku")
    channel_info = read_meta_info(analysis_defn, "channel")
    mainprogram_path = get_mainprogram_path(
        analysis_defn.get_mainprogram_date()
    )

    sheet_headers = pl.read_excel(
        mainprogram_path,
        sheet_name=NA_FBA_SHEET,
        read_options={
            "header_row": 1,
            "n_rows": 0,
        },
    )

    use_columns = []
    intermediate_names = []
    for col in sheet_headers.columns:
        for x in ["merchant", ["qty", "box"]]:
            if all([y in col.lower() for y in normalize_as_list(x)]):
                if x == "merchant":
                    intermediate_names.append("sku")
                else:
                    intermediate_names.append("qty_box")
                use_columns.append(col)

    raw_df = pl.read_excel(
        mainprogram_path,
        sheet_name=NA_FBA_SHEET,
        read_options={
            "header_row": 1,
            "use_columns": use_columns,
            "skip_rows": 1,
        },
    ).rename(dict(zip(use_columns, intermediate_names, strict=True)))
    # sys.displayhook(raw_df)

    qty_box_df = cast_standard(
        [active_sku_info, channel_info],
        sanitize_excel_extraction(raw_df),
        strict=False,
    ).drop_nulls("sku")

    # sys.displayhook(find_dupes(qty_box_df, ["sku"]))
    # sys.displayhook(
    #     find_dupes(active_sku_info.select(Sku.members(Priority.META)), ["sku"])
    # )
    qty_box_df = (
        # when SKUs are changed manually in the main program in order to resolve
        # issues arounding m_sku and a_sku not matching, then we end up having
        # duplicates for qty_box info, so unique them out
        qty_box_df.unique()
        .join(
            active_sku_info.select(Sku.members(MemberType.META)),
            on="sku",
            # we expect 1:1 match on SKUs
            validate="1:1",
            nulls_equal=True,
        )
        .cast({"qty_box": pl.Int64()})
        .drop_nulls()
        .unique()
    )

    write_df(overwrite, save_path, qty_box_df)

    return qty_box_df
