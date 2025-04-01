"""Functions to read from the Master SKU information Excel file."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
import sys
import polars as pl
import polars.selectors as cs
import re

from jjpred.analysisdefn import AnalysisDefn
from jjpred.countryflags import CountryFlags
from jjpred.globalpaths import (
    ANALYSIS_INPUT_FOLDER,
    ANALYSIS_OUTPUT_FOLDER,
)
from jjpred.readsupport.inventory import InventoryType, read_inventory
from jjpred.sku import Sku
from jjpred.seasons import POSeason, Season
from jjpred.structlike import MemberType
from jjpred.utils.datetime import Date, DateLike
from jjpred.utils.fileio import (
    delete_or_read_df,
    disable_fastexcel_dtypes_logger,
    write_df,
)
from jjpred.utils.polars import (
    binary_partition_strict,
    binary_partition_weak,
    convert_dict_to_polars_df,
    find_dupes,
    sanitize_excel_extraction,
)
from jjpred.utils.typ import RuntimeCheckableDataclass


def column_renamer(x: str) -> str:
    """Renames columns as relevant for the Master SKU file."""
    x = re.sub(r"\s+", "_", x.lower())
    if "fbasku" in x:
        return f"fba_sku_{x.split('_')[-1]}"

    for compare_return_pair in [
        ("seasons_sku", "season_history"),
        "category",
        "status",
        ("print_sku", "print"),
        ("print_name", "print_name"),
        "size",
        ("msku", "m_sku"),
        ("adjust", "a_sku"),
        ("pause_plan", "pause_plan_str"),
    ]:
        if isinstance(compare_return_pair, tuple):
            compare, return_str = compare_return_pair
        else:
            compare, return_str = compare_return_pair, compare_return_pair

        if compare in x:
            return return_str

    return x


def parse_pause_plan(pause_plan: str) -> CountryFlags:
    """Parse a pause plan string as given in the Master SKU file."""
    pause_plan = re.sub(r"\s+", " ", pause_plan).lower()
    split_parts = pause_plan.split(" ", 1)
    if split_parts[0] == "all" and split_parts[1] == "active":
        return CountryFlags(0)
    elif "undefined" in split_parts[0]:
        return CountryFlags.all_regions()
    else:
        result = CountryFlags(0)
        for str_country in split_parts[0].split("/"):
            result |= CountryFlags.from_str(str_country)

        return result


@dataclass
class MasterSkuInfo(RuntimeCheckableDataclass):
    """Data from the Master SKU file."""

    active_sku: pl.DataFrame = field(init=False)
    """Active SKUs."""
    fba_sku: pl.DataFrame = field(init=False)
    """FBA SKUs associated with active SKUs."""
    all_sku: pl.DataFrame = field(init=False)
    """Includes inactive SKUs ("inactive", "retired", etc.), but not ignored
    SKUs."""
    print_name_map: pl.DataFrame = field(init=False)
    """Print names taken from the Master SKU file, mapped in some cases to adjusted
names because they are otherwise unnecessarily long."""
    numeric_size_map: pl.DataFrame = field(init=False)
    """Sizes mapped to numeric values. Useful for sorting by size especially when
plotting."""
    ignored_sku: pl.DataFrame = field(default_factory=pl.DataFrame, init=False)
    """Ignored SKUs."""

    @classmethod
    def generate_save_paths(
        cls, analysis_defn: AnalysisDefn
    ) -> dict[str, Path]:
        """Generate paths where Master SKU information should be saved to."""
        website_sku_date = analysis_defn.get_website_sku_date()

        if website_sku_date is not None:
            websku_str = f"_WEBSKU_{website_sku_date}"
        else:
            websku_str = ""

        return dict(
            (
                df_name,
                ANALYSIS_OUTPUT_FOLDER.joinpath(
                    f"{analysis_defn.tag()}_master_sku_{websku_str}{df_name}"
                    ".parquet",
                ),
            )
            for df_name in MasterSkuInfo.fields()
        )

    @classmethod
    def delete_or_read(
        cls,
        analysis_defn: AnalysisDefn,
        delete_if_exists: bool = False,
    ) -> MasterSkuInfo | None:
        """Read master SKU information associated with an analysis, or delete
        such information if it exists."""
        field_dfs = {}
        save_paths = MasterSkuInfo.generate_save_paths(analysis_defn)
        for f, save_path in save_paths.items():
            field_dfs[f] = delete_or_read_df(delete_if_exists, save_path)
        result = MasterSkuInfo()

        for f, df in field_dfs.items():
            if df is None:
                return None
            else:
                result.__setattr__(f, df)

        return result

    def write_to_disk(self, analysis_defn: AnalysisDefn) -> None:
        """Write master SKU information dataframes to disk."""
        save_paths = MasterSkuInfo.generate_save_paths(analysis_defn)
        for f, save_path in save_paths.items():
            write_df(True, save_path, self.__getattribute__(f))

    @classmethod
    def get_master_sku_info(
        cls,
        analysis_defn: AnalysisDefn,
        read_from_disk=False,
        delete_if_exists=False,
        # website_sku_fetch_info: WebsiteSkuFetchInfo | None = None,
    ) -> MasterSkuInfo:
        master_sku_info = None
        if read_from_disk or delete_if_exists:
            master_sku_info = MasterSkuInfo.delete_or_read(
                analysis_defn, delete_if_exists
            )

        if master_sku_info is not None:
            return master_sku_info
        else:
            website_sku_date = analysis_defn.get_website_sku_date()
            if website_sku_date is not None:
                website_sku_fetch_info = WebsiteSkuFetchInfo(
                    website_sku_date,
                    read_inventory(
                        analysis_defn,
                        InventoryType.AUTO,
                        read_from_disk=read_from_disk,
                        delete_if_exists=delete_if_exists,
                    ),
                )
            else:
                website_sku_fetch_info = None

            master_sku_info = read_master_sku_excel_file(
                analysis_defn.master_sku_date,
                website_sku_fetch_info,
            )

            master_sku_info.write_to_disk(analysis_defn)

            return master_sku_info


def filter_m_and_u_types(all_sku_info: pl.DataFrame) -> pl.DataFrame:
    return all_sku_info.filter(
        ~(
            (
                pl.col("sku")
                .cast(pl.String())
                .str.split("-")
                .list.first()
                .str.ends_with("U")
                & pl.col("sku")
                .cast(pl.String())
                .str.split("-")
                .list.first()
                .str.len_chars()
                .gt(3)
            )
            | (
                pl.col("sku")
                .cast(pl.String())
                .str.split("-")
                .list.first()
                .str.starts_with("M")
                & pl.col("sku")
                .cast(pl.String())
                .str.split("-")
                .list.first()
                .str.len_chars()
                .gt(3)
            )
        )
    )


MASTER_SKU_DF_HEADER_ROW: int = 3
"""The header row of the Master SKU file (0-based index)."""

ADJUSTED_PRINT_NAMES: dict[str, str] = {
    "Bear w/Cream Lining": "Bear, Cream Lining",
    "Bear w/Grey Lining": "Bear, Grey Lining",
    "Black w/ Black Trim": "Black, Black Trim",
    "Black w/Cream Lining": "Black, Cream Lining",
    "Black w/Grey Lining": "Black, Grey Lining",
    "Blue w/ self trim": "Blue, self trim",
    "Bunny w/Cream Lining": "Bunny, Cream Lining",
    "Bunny w/Grey Lining": "Bunny, Grey Lining",
    "Cream (Twinning Beanie)": "Cream",
    "Grey (Twinning Beanie)": "Grey",
    "Grey w/Cream Lining": "Grey, Cream Lining",
    "Grey w/Grey Lining": "Grey, Grey Lining",
    # "Large  Herringbone": "Large Herringbone",
    "Mustard (Twinning Beanie)": "Mustard",
    "Navy w/ Navy Trim": "Navy, Navy Trim",
    "Pink / Pale Pink (knit shoes)": "Pink/Pale Pink",
    "Purple  (lighter purple)": "Lighter Purple",
    "Purple w/Cream Lining": "Purple, Cream Lining",
    "Purple w/Grey Lining": "Purple, Grey Lining",
    "Rainbow w/ Pink Trim": "Rainbow, Pink Trim",
    "Shark w/ Navy Trim": "Shark, Navy Trim",
}
"""Print names taken from the Master SKU file, mapped in some cases to adjusted
names because they are otherwise unnecessarily long."""

NUMERIC_SIZE: dict[str, float] = {
    "6m": 0.5,
    "12m": 1,
    "18m": 1.5,
    "1T": 1,
    "2T": 2,
    "3T": 3,
    "4E": 4,
    "4T": 4,
    "5E": 5,
    "5T": 5,
    "6E": 6,
    "6Y": 6,
    "7T": 7,
    "7Y": 7,
    "8T": 8,
    "8Y": 8,
    "10Y": 10,
    "12Y": 12,
    "14Y": 14,
    "16Y": 16,
    "XS": 1,
    "S": 2,
    "S & M": 2.5,
    "M": 3,
    "L": 4,
    "XL": 5,
    "XXL": 6,
    "OS": 100,
    "20E": 20,
    "21E": 21,
    "22E": 22,
    "JR1": 14,  # JR1 is one size larger than 13
    "JR2": 15,
    "JR3": 16,
    "JR4": 17,
}
"""Sizes mapped to numeric values. Useful for sorting by size especially when
plotting. Sizes that do not appear here are either: 1) easily parsed into a
numeric value (e.g. size "10.5" can be straightforwardly parsed into ``10.5``),
or 2) have not yet been given a mapping.."""


def generate_filtered_season_history_map(
    master_sku_df: pl.DataFrame,
) -> pl.DataFrame:
    """Filter out TAIWAN, CLEMENT, and other issues."""
    unique_season_histories = (
        master_sku_df.select("season_history")
        .unique()
        .with_columns(
            season_history_tags=pl.col.season_history.str.split(",")
            .list.eval(pl.element().str.split(" "))
            .list.eval(pl.element().explode())
            .list.eval(pl.element().filter(pl.element().str.len_chars().gt(0)))
        )
    )

    season_history_tag_map = (
        unique_season_histories.select("season_history_tags")
        .explode("season_history_tags")
        .rename({"season_history_tags": "season_history_tag"})
        .unique()
        .with_columns(
            pl.col.season_history_tag.str.split("-")
            .list.eval(pl.element().str.strip_chars())
            .alias("season_history_tag_parts")
        )
        .with_columns(
            main_season_tag=pl.col.season_history_tag_parts.list.first(),
            season_tag_extra=pl.col.season_history_tag_parts.list.slice(
                1
            ).list.eval(pl.element().str.to_uppercase()),
        )
        .drop("season_history_tag_parts")
        .with_columns(
            season_year_remainder=pl.col.main_season_tag.str.extract_groups(
                r"^(?P<year>\d+)(?P<po_season>F|S)?(?P<remainder>.*)"
            ).cast(
                pl.Struct(
                    {
                        "year": pl.Int64(),
                        "po_season": POSeason.polars_type(),
                        "remainder": pl.String(),
                    }
                )
            )
        )
        .unnest("season_year_remainder")
        .with_columns(
            remainder=pl.when(
                pl.col.remainder.is_not_null()
                & pl.col.remainder.str.len_chars().eq(0)
            )
            .then(None)
            .otherwise(pl.col.remainder)
        )
    )

    season_history_tag_map, non_empty_extras = binary_partition_strict(
        season_history_tag_map, ~pl.col.season_tag_extra.list.len().gt(0)
    )

    season_history_tag_map, other_problems = binary_partition_strict(
        season_history_tag_map,
        pl.col.po_season.is_not_null() & pl.col.year.is_not_null(),
    )

    recoverable, non_recoverable = binary_partition_weak(
        other_problems,
        pl.col.year.eq(22)
        & pl.col.po_season.is_null()
        & pl.col.remainder.eq("D"),
    )

    recoverable = recoverable.with_columns(
        po_season=pl.lit(["SS", "FW"], dtype=pl.List(POSeason.polars_type()))
    ).explode("po_season")

    season_history_tag_map = season_history_tag_map.vstack(
        recoverable.select(season_history_tag_map.columns)
    )

    # assert non_recoverable is None or len(non_recoverable) == 0, (
    #     non_recoverable
    # )

    season_history_map = (
        unique_season_histories.explode("season_history_tags")
        .unique()
        .rename({"season_history_tags": "season_history_tag"})
        .join(season_history_tag_map, on=["season_history_tag"])
        .select(
            "season_history",
            pl.struct(
                "year",
                "po_season",
                # "remainder", "season_tag_extra"
            ).alias(
                "season_history_info",
            ),
        )
        .group_by("season_history")
        .agg(pl.col.season_history_info.unique())
        .filter(pl.col.season_history_info.list.len().gt(0))
        .with_columns(
            sku_year_history=pl.col.season_history_info.list.eval(
                pl.element().struct.field("year")
            ).list.sort(descending=True)
        )
        .with_columns(sku_latest_year=pl.col.sku_year_history.list.first())
    )

    sku_latest_po_season = (
        season_history_map.explode("season_history_info")
        .filter(
            pl.col("season_history_info")
            .struct.field("year")
            .eq(pl.col("sku_latest_year"))
        )
        .with_columns(
            sku_latest_po_season=pl.col.season_history_info.struct.field(
                "po_season"
            )
        )
        .group_by("season_history")
        .agg(pl.col.sku_latest_po_season.sort().first())
    )

    season_history_map = season_history_map.join(
        sku_latest_po_season, on=["season_history"], how="left"
    )

    assert (
        len(season_history_map.filter(pl.col.sku_latest_po_season.is_null()))
        == 0
    )

    return season_history_map


def read_raw_master_sku(master_sku_date: DateLike) -> pl.DataFrame:
    disable_fastexcel_dtypes_logger()

    master_sku_path = ANALYSIS_INPUT_FOLDER.joinpath(
        Path(
            f"1-MasterSKU-All-Product-{Date.from_datelike(master_sku_date).strftime('%Y-%m-%d')}.xlsx"
        )
    )

    assert master_sku_path.is_file(), master_sku_path
    header_df = pl.read_excel(
        master_sku_path,
        sheet_name="MasterFile",
        read_options={"header_row": MASTER_SKU_DF_HEADER_ROW, "n_rows": 0},
    ).rename(lambda x: re.sub(r"\s+|_x000D_", " ", x))

    required_columns = [
        ix
        for ix, x in enumerate(header_df.columns)
        if (
            (
                x.lower()
                in [
                    "msku status",
                    "seasons sku",
                    "print sku",
                    "print name",
                    "category sku",
                    "size",
                    "msku",
                ]
            )
            or any([y in x.lower() for y in ["fbasku", "adjust", "pause"]])
        )
    ]

    master_sku_df = sanitize_excel_extraction(
        pl.read_excel(
            master_sku_path,
            sheet_name="MasterFile",
            read_options={
                "header_row": MASTER_SKU_DF_HEADER_ROW,
                "use_columns": required_columns,
            },
        )
        .rename(lambda x: re.sub(r"\s+|_x000D_", " ", x))
        .rename(column_renamer)
        .with_columns(status=pl.col("status").str.to_lowercase())
    )

    # we assume that `sku` (aka `m_sku`) is unique
    assert len(master_sku_df) == len(master_sku_df["m_sku"].unique())

    return master_sku_df


def read_website_sku_list(
    website_sku_list_date: DateLike,
) -> pl.DataFrame:
    website_sku_path = ANALYSIS_INPUT_FOLDER.joinpath(
        Path(f"wc-sku-{Date.from_datelike(website_sku_list_date)}.csv")
    )

    disable_fastexcel_dtypes_logger()
    website_sku = pl.read_csv(website_sku_path)

    all_website_skus = website_sku.select(pl.col("SKU").alias("sku")).filter(
        pl.col.sku.is_not_null()
    )

    return all_website_skus


@dataclass
class WebsiteSkuFetchInfo:
    website_sku_date: DateLike
    inventory_df: pl.DataFrame

    def __init__(
        self, website_sku_date: DateLike | None, inventory_df: pl.DataFrame
    ):
        if website_sku_date is not None:
            self.website_sku_date = website_sku_date
        else:
            raise ValueError(f"{website_sku_date=}")
        self.inventory_df = inventory_df


def get_relevant_website_sku(
    website_sku_fetch_info: WebsiteSkuFetchInfo,
    # master_sku_date_or_df: DateLike | pl.DataFrame,
) -> pl.DataFrame:
    # if isinstance(master_sku_date_or_df, pl.DataFrame):
    #     master_sku_df = master_sku_date_or_df
    # else:
    #     master_sku_df = read_raw_master_sku(master_sku_date_or_df)

    all_website_skus = read_website_sku_list(
        website_sku_fetch_info.website_sku_date
    )
    inventory_df = website_sku_fetch_info.inventory_df

    find_dupes(all_website_skus, ["sku"], raise_error=True)
    find_dupes(inventory_df, ["sku"], raise_error=True)

    relevant_inventory = inventory_df.filter(
        pl.col.channel.eq("Warehouse CA WAREHOUSE")
    ).filter(pl.col.stock.is_not_null() & pl.col.stock.gt(3))

    relevant_website_skus = all_website_skus.join(
        relevant_inventory.select("sku"), on="sku"
    )

    return relevant_website_skus


# def get_relevant_website_sku(
#     website_sku_list_date: DateLike,
#     master_sku_date_or_df: DateLike | pl.DataFrame,
# ) -> pl.DataFrame:
#     if isinstance(master_sku_date_or_df, pl.DataFrame):
#         master_sku_df = master_sku_date_or_df
#     else:
#         master_sku_df = read_raw_master_sku(master_sku_date_or_df)

#     website_sku_list = read_website_sku_list(website_sku_list_date)

#     relevant_sku = (
#         website_sku_list.join(master_sku_df.rename({"m_sku": "sku"}), on="sku")
#         .select("sku")
#         .unique()
#         .sort("sku")
#     )

#     return relevant_sku


def read_master_sku_excel_file(
    master_sku_date: DateLike,
    website_sku_fetch_info: WebsiteSkuFetchInfo | None,
) -> MasterSkuInfo:
    """Read the master information excel file."""

    master_sku_df = read_raw_master_sku(master_sku_date)

    if website_sku_fetch_info is not None:
        website_sku = get_relevant_website_sku(website_sku_fetch_info).rename(
            {"sku": "m_sku"}
        )
    else:
        website_sku = None

    if website_sku is not None:
        website_skus_missing_from_master_sku = (
            website_sku.join(
                master_sku_df.select("m_sku").unique(),
                on=["m_sku"],
                how="anti",
            )
            .with_columns(
                # ^(?P<category>MBH|MCH|MKB|SKB-INSOL)-(?P<print>[^-]+)-(?P<size>[^-]+)(?:-(?P<remainder>.*))?$
                pl.col.m_sku.str.extract_groups(
                    r"^(?P<category>MBH|MCH|MKB|SKB)-(?P<print>[^-]+)-(?P<size>[^-]+)"
                ).alias("extracted"),
                status=pl.lit("active"),
                a_sku=pl.col.m_sku,
                print_name=pl.lit(""),
                website_sku=pl.lit(True),
                orphan_sku=pl.lit(True),
                pause_plan_str=pl.lit("<undefined>"),
            )
            .unnest("extracted")
            .with_columns(
                category=pl.when(pl.col.m_sku.str.starts_with("SKB-INSOL"))
                .then(pl.lit("SKB-INSOL"))
                .otherwise(pl.col.category)
            )
            # .with_columns(
            #     size=pl.when(pl.col.category.eq("SKB-INSOL"))
            #     .then(pl.col.print)
            #     .otherwise(pl.col.size),
            #     print=pl.when(pl.col.category.eq("SKB-INSOL"))
            #     .then(pl.col.size)
            #     .otherwise(pl.col.print),
            # )
        )

        if (
            len(
                website_skus_missing_from_master_sku.filter(
                    pl.col.category.is_null()
                    | pl.col.print.is_null()
                    | pl.col.size.is_null()
                )
            )
            > 0
        ):
            pl.Config().set_tbl_rows(-1)
            print(
                website_skus_missing_from_master_sku.filter(
                    pl.col.category.is_null()
                    | pl.col.print.is_null()
                    | pl.col.size.is_null()
                ).select("m_sku")
            )
            pl.Config().restore_defaults()
            raise ValueError("Found problematic extra SKUs")

        dummy_master_sku_info = pl.DataFrame(schema=master_sku_df.schema)
        additional_master_sku_info = dummy_master_sku_info.select(
            cs.exclude(
                [
                    x
                    for x in website_skus_missing_from_master_sku.columns
                    if x != "m_sku"
                ]
            )
        ).join(
            website_skus_missing_from_master_sku,
            how="full",
            on="m_sku",
            coalesce=True,
        )

        master_sku_with_website_info = master_sku_df.join(
            website_sku.with_columns(
                website_sku=pl.lit(True), orphan_sku=pl.lit(False)
            ),
            on=["m_sku"],
            how="left",
        )

        master_sku_df = pl.concat(
            [
                master_sku_with_website_info,
                additional_master_sku_info.select(
                    master_sku_with_website_info.columns
                ),
            ]
        ).with_columns(
            pl.col.website_sku.fill_null(pl.lit(False)),
            pl.col.orphan_sku.fill_null(pl.lit(False)),
        )
    # print(master_sku_df["status"].unique().sort())

    current_year_integer = Date.from_datelike(master_sku_date).year - 2000
    master_sku_df = master_sku_df.with_columns(
        status=pl.col("status").str.to_lowercase().fill_null("inactive"),
    ).with_columns(
        season_history=pl.when(
            pl.col.season_history.eq("") | pl.col.season_history.is_null()
        )
        .then(pl.lit(f"{current_year_integer}S,{current_year_integer}F"))
        .otherwise(pl.col.season_history)
    )

    master_sku_df = master_sku_df.filter(
        pl.col("status").is_in(
            [
                "active",
                "inactive",
                "retired",
                "discontinued",
                "cancelled",
                "deleted",
            ]
        )
    )

    master_sku_df = (
        master_sku_df.with_columns(
            m_sku_parsed=pl.col("m_sku")
            .map_elements(
                Sku.map_polars,
                return_dtype=Sku.intermediate_polars_type_struct(),
            )
            .struct.rename_fields(["m_" + k for k in Sku.members()])
        )
        .unnest("m_sku_parsed")
        .with_columns(sku_remainder=pl.col.m_sku_remainder)
        # .drop("m_print", "m_size")
        # .rename({"m_sku_remainder": "sku_remainder"})
        .with_columns(
            a_sku_parsed=pl.col("a_sku")
            .map_elements(
                Sku.map_polars,
                return_dtype=Sku.intermediate_polars_type_struct(),
            )
            .struct.rename_fields(["a_" + k for k in Sku.members()])
        )
        .unnest("a_sku_parsed")
        # .drop("a_print", "a_size", "a_sku_remainder")
    )

    # === mark special categories ===
    # unique_categories = reduce(
    #     lambda x, y: x.extend(y),
    #     [master_sku_df[f"{y}category"].unique() for y in ["", "a_", "m_"]],
    # ).unique()
    # is_special_filter = reduce(
    #     lambda x, y: x | y,
    #     [
    #         (
    #             pl.col(v).str.starts_with("M")
    #             & (
    #                 pl.col(v).str.slice(1).is_in(unique_categories)
    #                 | ~pl.col(v).is_in(STARTS_WITH_M_OK_LIST)
    #             )
    #         )
    #         | (
    #             pl.col(v).str.ends_with("U")
    #             & (
    #                 pl.col(v)
    #                 .str.slice(0, length=pl.col(v).str.len_chars() - 1)
    #                 .is_in(unique_categories)
    #                 | ~pl.col(v).is_in(ENDS_WITH_U_OK_LIST)
    #             )
    #         )
    #         | pl.col(v).is_in(IGNORE_CATEGORY_LIST)
    #         for v in ["category", "a_category", "m_category"]
    #     ],
    # )
    # print("Master SKU will ignore:")
    # will_ignore = (
    #     master_sku_df.filter(is_special_filter)
    #     .select("m_sku", "a_sku", "m_category", "a_category", "category")
    #     .rename({"m_sku": "sku"})
    #     .unique()
    #     .sort("category")
    # )
    # pl.Config.set_tbl_rows(len(will_ignore))
    # sys.displayhook(
    #     will_ignore.select(cs.exclude("sku", "a_sku"))
    #     .unique()
    #     .sort("category")
    # )
    # pl.Config.restore_defaults()
    # master_sku_df = (
    #     master_sku_df.filter(~is_special_filter)
    #     # master_sku_df.filter(~is_special_filter).drop(
    #     #   "a_category", "m_category"
    #     # )
    # )
    # === === ===

    # === casting columns as enum ===
    related_columns = {}
    for z in Sku.members(MemberType.META):
        related_columns[z] = [
            f"{y}{z}"
            for y in ["", "a_", "m_"]
            if f"{y}{z}" in master_sku_df.columns
        ]

    related_casting_columns = []
    for v in related_columns.values():
        related_casting_columns += v

    category_per_related_column = {}
    for c in related_casting_columns:
        for z in Sku.members(MemberType.META):
            if c in related_columns[z]:
                category_per_related_column[c] = z

    other_casting_columns = ["status", "print_name"]

    default_values_for_related = {}
    for z in Sku.members(MemberType.META):
        default_values_for_related[z] = Sku.field_defaults.get(z, None)

    unique_values_for_related = {}
    # sometimes, for analysis purposes, we might aggregate data related to a
    # particular column (e.g. aggregate all "size" sales data into "_ALL_")
    may_be_combined = [
        c for c in Sku.members(MemberType.SECONDARY) if c != "category"
    ]
    for z in Sku.members(MemberType.META):
        unique_values = pl.Series(z, [], dtype=pl.String())
        for c in related_columns[z]:
            unique_values = unique_values.extend(
                master_sku_df[c].unique()
            ).unique()

        if z in may_be_combined:
            unique_values_for_related[z] = (
                unique_values.extend(
                    pl.Series(z, ["_ALL_"], dtype=pl.String())
                )
                .unique()
                .sort()
            )
        else:
            unique_values_for_related[z] = unique_values

    unique_values = {
        k: v.sort() for k, v in unique_values_for_related.items()
    } | {k: master_sku_df[k].unique().sort() for k in other_casting_columns}

    dtypes: Mapping[str, pl.DataType] = {
        k: pl.Enum(unique_values[category_per_related_column[k]].sort())
        for k in related_casting_columns
        if k in master_sku_df.columns
    } | {
        k: pl.Enum(unique_values[k].sort())
        for k in other_casting_columns
        if k in master_sku_df.columns
    }
    master_sku_df = master_sku_df.cast(dtypes)  # type: ignore
    # === === ==

    # === parse pause plans ==
    unique_plans = pl.Enum(master_sku_df["pause_plan_str"].unique().sort())

    master_sku_df = (
        master_sku_df.cast({"pause_plan_str": unique_plans})
        .join(
            pl.DataFrame(
                [
                    pl.Series(
                        "pause_plan_str",
                        unique_plans.categories,
                        dtype=unique_plans,
                    )
                ]
            ).with_columns(
                pause_plan=pl.col("pause_plan_str").map_elements(
                    parse_pause_plan, return_dtype=pl.Int64()
                )
            ),
            on="pause_plan_str",
            # we are doing "map_elements" using a join, so we expect m:1
            validate="m:1",
            nulls_equal=True,
        )
        .drop("pause_plan_str")
    )
    # === === ==

    # Sometimes "season history" will contain
    # information such as `24F-Taiwan`, or `24-Taiwan`, and in general, one can
    # expect it to not remain of a fixed format. Therefore, filter out all such
    # variants apart from the ones we want to explicitly recognize: F*, S*

    season_history_map = generate_filtered_season_history_map(master_sku_df)

    master_sku_df = master_sku_df.join(
        season_history_map, on=["season_history"]
    )

    master_sku_df = (
        master_sku_df.with_columns(
            sku_year_history=pl.col.season_history_info.list.eval(
                pl.element().struct.field("year")
            )
            .list.unique()
            .list.drop_nulls()
            .list.sort(descending=True)
        )
        .with_columns(
            sku_latest_year=pl.col("sku_year_history")
            .list.drop_nulls()
            .list.first()
        )
        .filter(pl.col("sku_latest_year").is_not_null())
    )

    master_sku_df = master_sku_df.join(
        master_sku_df.select("category", "sku_year_history")
        .unique()
        .group_by("category")
        .agg(
            pl.col("sku_year_history")
            .flatten()
            .alias("category_year_history"),
        )
        .with_columns(
            pl.col("category_year_history").list.unique().list.sort()
        ),
        on="category",
        how="left",
        validate="m:1",
    )
    # === === ==

    sku_columns = Sku.members(MemberType.SECONDARY) + ["a_sku", "m_sku"]

    latest_po_seasons_per_category = (
        (
            master_sku_df
            # .filter(pl.col.status.eq("active"))
            .explode("season_history_info")
            .select("season_history_info", "sku_latest_year", *sku_columns)
            .filter(
                pl.col("season_history_info")
                .struct.field("year")
                .eq(pl.col("sku_latest_year"))
                | pl.col("season_history_info")
                .struct.field("year")
                .eq(pl.col("sku_latest_year") - 1)
            )
            .with_columns(
                latest_po_season=pl.col("season_history_info").struct.field(
                    "po_season"
                )
            )
        )
        .with_columns(
            is_fw=pl.col.latest_po_season.eq("FW"),
            is_ss=pl.col.latest_po_season.eq("SS"),
        )
        .group_by(pl.col.category)
        .agg(
            pl.col.is_fw.sum().alias("fw_tally"),
            pl.col.is_ss.sum().alias("ss_tally"),
            pl.col.latest_po_season.alias("latest_po_seasons"),
        )
        .with_columns(total_tally=pl.col.fw_tally + pl.col.ss_tally)
        .with_columns(
            tally_fraction=pl.col.fw_tally / pl.col.total_tally,
            latest_po_seasons=pl.col.latest_po_seasons.list.unique()
            .list.eval(pl.element().cast(pl.String()))
            .list.sort()
            .list.join(","),
        )
    ).with_columns(
        # we manually inspected "tally_fraction", and then determined that the
        # following categories should be marked FW, even though they
        # have some PO that happened in SS
        #
        # later on, we just read from the CONFIG file to determine seasonality
        # of items
        latest_po_seasons=pl.when(pl.col.category.is_in(["WSF", "WJT", "IHT"]))
        .then(pl.lit("FW"))
        .otherwise(pl.col.latest_po_seasons)
    )

    season_map = latest_po_seasons_per_category.join(
        latest_po_seasons_per_category.select("latest_po_seasons")
        .unique()
        .with_columns(
            season=pl.col("latest_po_seasons")
            .map_elements(Season.map_polars, return_dtype=pl.String())
            .cast(Season.polars_type())
        ),
        on="latest_po_seasons",
        how="left",
        # there are multiple SKU with the same cat in the LHS
        validate="m:1",
        nulls_equal=True,
    ).select("category", "season")

    master_sku_df = (
        master_sku_df.join(
            season_map,
            on="category",
            # there are many SKU with the same category in the LHS
            validate="m:1",
            nulls_equal=True,
        )
        # .drop("season_history_info")
        .cast(
            {
                "season_history": pl.Enum(
                    master_sku_df["season_history"].unique().sort()
                )
            }
        )
    )

    master_sku_df = master_sku_df.rename({"m_sku": "sku"})

    fba_sku = master_sku_df.select(
        ["a_sku", "sku"]
        + list(cs.expand_selector(master_sku_df, cs.contains("fba_sku")))
    )
    fba_sku = (
        fba_sku.unpivot(
            index=["a_sku", "sku"],
            on=cs.contains("fba_sku"),
            variable_name="country",
            value_name="fba_sku",
        )
        .drop_nulls()
        .with_columns(pl.col("country").str.split("fba_sku_").list.last())
    )
    fba_sku = fba_sku.cast(
        {"country": pl.Enum(fba_sku["country"].drop_nulls().unique().sort())}
    )
    country_flag = (
        fba_sku.select("country")
        .unique()
        .sort("country")
        .with_columns(
            country_flag=pl.col("country").map_elements(
                lambda x: CountryFlags.from_str(x).value,
                return_dtype=pl.Int64(),
            )
        )
    )
    fba_sku = fba_sku.join(
        country_flag,
        on="country",
        how="left",
        validate="m:1",
        nulls_equal=True,
    ).drop("country")

    assert (
        len(
            master_sku_df.select("print", "print_name")
            .group_by("print")
            .agg(pl.col.print_name.unique())
            .filter(pl.col.print_name.list.len().gt(1))
        )
        == 0
    )

    print_name_map = (
        master_sku_df.select("print", "print_name")
        .unique()
        .join(
            convert_dict_to_polars_df(
                ADJUSTED_PRINT_NAMES, "print_name", "adjusted_print_name"
            ).cast({"print_name": master_sku_df["print_name"].dtype}),
            how="left",
            on="print_name",
        )
        .with_columns(
            adjusted_print_name=pl.when(pl.col.adjusted_print_name.is_null())
            .then(pl.col.print_name.cast(pl.String()))
            .otherwise(pl.col.adjusted_print_name)
        )
    )
    print_name_map = (
        print_name_map.cast(
            {
                "adjusted_print_name": pl.Enum(
                    print_name_map["adjusted_print_name"].unique().sort()
                )
            }
        )
        .drop("print_name")
        .rename({"adjusted_print_name": "print_name"})
    )

    numeric_size_map = (
        master_sku_df.select("size")
        .unique()
        .join(
            convert_dict_to_polars_df(
                NUMERIC_SIZE, "size", "numeric_size"
            ).cast(
                {
                    "size": master_sku_df["size"].dtype,
                    "numeric_size": pl.Float32(),
                }
            ),
            on=["size"],
            how="left",
        )
        .with_columns(
            numeric_size=pl.when(pl.col.numeric_size.is_null())
            .then(pl.col.size.cast(pl.Float32(), strict=False))
            .otherwise(pl.col.numeric_size)
        )
    )

    problem_sizes = numeric_size_map.filter(pl.col.numeric_size.is_null())
    if len(problem_sizes) > 0:
        sys.displayhook(problem_sizes)
        raise ValueError("Some sizes are not mapped to a numeric size!")

    master_sku_df = master_sku_df.select(cs.exclude(cs.contains("fba_sku")))
    result = MasterSkuInfo()
    result.active_sku = master_sku_df.filter(
        pl.col("status").eq("active")
    ).drop("status")
    result.all_sku = master_sku_df.select(cs.exclude(cs.contains("fba_sku")))
    result.fba_sku = fba_sku
    result.ignored_sku = pl.DataFrame(schema=master_sku_df.schema)
    result.print_name_map = print_name_map
    result.numeric_size_map = numeric_size_map

    return result


# def read_master_sku_excel_file_old(master_sku_date: DateLike) -> MasterSkuInfo:
#     """Read the master information excel file."""

#     master_sku_df = read_raw_master_sku(master_sku_date)

#     master_sku_df = master_sku_df.with_columns(
#         status=pl.col("status").str.to_lowercase().fill_null("NO_STATUS"),
#         season_history=pl.col("season_history").fill_null(""),
#     )
#     # print(master_sku_df["status"].unique().sort())

#     master_sku_df = master_sku_df.filter(
#         pl.col("status").is_in(
#             [
#                 "active",
#                 "inactive",
#                 "retired",
#                 "discontinued",
#                 "cancelled",
#                 "deleted",
#             ]
#         )
#     )

#     master_sku_df = (
#         master_sku_df.with_columns(
#             m_sku_parsed=pl.col("m_sku")
#             .map_elements(
#                 Sku.map_polars,
#                 return_dtype=Sku.intermediate_polars_type_struct(),
#             )
#             .struct.rename_fields(["m_" + k for k in Sku.members()])
#         )
#         .unnest("m_sku_parsed")
#         .with_columns(sku_remainder=pl.col.m_sku_remainder)
#         # .drop("m_print", "m_size")
#         # .rename({"m_sku_remainder": "sku_remainder"})
#         .with_columns(
#             a_sku_parsed=pl.col("a_sku")
#             .map_elements(
#                 Sku.map_polars,
#                 return_dtype=Sku.intermediate_polars_type_struct(),
#             )
#             .struct.rename_fields(["a_" + k for k in Sku.members()])
#         )
#         .unnest("a_sku_parsed")
#         # .drop("a_print", "a_size", "a_sku_remainder")
#     )

#     # === mark special categories ===
#     unique_categories = reduce(
#         lambda x, y: x.extend(y),
#         [master_sku_df[f"{y}category"].unique() for y in ["", "a_", "m_"]],
#     ).unique()
#     is_special_filter = reduce(
#         lambda x, y: x | y,
#         [
#             (
#                 pl.col(v).str.starts_with("M")
#                 & (
#                     pl.col(v).str.slice(1).is_in(unique_categories)
#                     | ~pl.col(v).is_in(STARTS_WITH_M_OK_LIST)
#                 )
#             )
#             | (
#                 pl.col(v).str.ends_with("U")
#                 & (
#                     pl.col(v)
#                     .str.slice(0, length=pl.col(v).str.len_chars() - 1)
#                     .is_in(unique_categories)
#                     | ~pl.col(v).is_in(ENDS_WITH_U_OK_LIST)
#                 )
#             )
#             | pl.col(v).is_in(IGNORE_CATEGORY_LIST)
#             for v in ["category", "a_category", "m_category"]
#         ],
#     )
#     print("Master SKU will ignore:")
#     will_ignore = (
#         master_sku_df.filter(is_special_filter)
#         .select("m_sku", "a_sku", "m_category", "a_category", "category")
#         .rename({"m_sku": "sku"})
#         .unique()
#         .sort("category")
#     )
#     pl.Config.set_tbl_rows(len(will_ignore))
#     sys.displayhook(
#         will_ignore.select(cs.exclude("sku", "a_sku"))
#         .unique()
#         .sort("category")
#     )
#     pl.Config.restore_defaults()
#     master_sku_df = (
#         master_sku_df.filter(~is_special_filter)
#         # master_sku_df.filter(~is_special_filter).drop(
#         #   "a_category", "m_category"
#         # )
#     )
#     # === === ===

#     # === casting columns as enum ===
#     related_columns = {}
#     for z in Sku.members(MemberType.META):
#         related_columns[z] = [
#             f"{y}{z}"
#             for y in ["", "a_", "m_"]
#             if f"{y}{z}" in master_sku_df.columns
#         ]

#     related_casting_columns = []
#     for v in related_columns.values():
#         related_casting_columns += v

#     category_per_related_column = {}
#     for c in related_casting_columns:
#         for z in Sku.members(MemberType.META):
#             if c in related_columns[z]:
#                 category_per_related_column[c] = z

#     other_casting_columns = ["status", "print_name"]

#     default_values_for_related = {}
#     for z in Sku.members(MemberType.META):
#         default_values_for_related[z] = Sku.field_defaults.get(z, None)

#     unique_values_for_related = {}
#     # sometimes, for analysis purposes, we might aggregate data related to a
#     # particular column (e.g. aggregate all "size" sales data into "_ALL_")
#     may_be_combined = [
#         c for c in Sku.members(MemberType.SECONDARY) if c != "category"
#     ]
#     for z in Sku.members(MemberType.META):
#         unique_values = pl.Series(z, [], dtype=pl.String())
#         for c in related_columns[z]:
#             unique_values = unique_values.extend(
#                 master_sku_df[c].unique()
#             ).unique()

#         if z in may_be_combined:
#             unique_values_for_related[z] = (
#                 unique_values.extend(
#                     pl.Series(z, ["_ALL_"], dtype=pl.String())
#                 )
#                 .unique()
#                 .sort()
#             )
#         else:
#             unique_values_for_related[z] = unique_values

#     unique_values = {
#         k: v.sort() for k, v in unique_values_for_related.items()
#     } | {k: master_sku_df[k].unique().sort() for k in other_casting_columns}

#     dtypes: Mapping[str, pl.DataType] = {
#         k: pl.Enum(unique_values[category_per_related_column[k]].sort())
#         for k in related_casting_columns
#         if k in master_sku_df.columns
#     } | {
#         k: pl.Enum(unique_values[k].sort())
#         for k in other_casting_columns
#         if k in master_sku_df.columns
#     }
#     master_sku_df = master_sku_df.cast(dtypes)  # type: ignore
#     # === === ==

#     # === parse pause plans ==
#     unique_plans = pl.Enum(master_sku_df["pause_plan_str"].unique().sort())

#     master_sku_df = (
#         master_sku_df.cast({"pause_plan_str": unique_plans})
#         .join(
#             pl.DataFrame(
#                 [
#                     pl.Series(
#                         "pause_plan_str",
#                         unique_plans.categories,
#                         dtype=unique_plans,
#                     )
#                 ]
#             ).with_columns(
#                 pause_plan=pl.col("pause_plan_str").map_elements(
#                     parse_pause_plan, return_dtype=pl.Int64()
#                 )
#             ),
#             on="pause_plan_str",
#             # we are doing "map_elements" using a join, so we expect m:1
#             validate="m:1",
#             nulls_equal=True,
#         )
#         .drop("pause_plan_str")
#     )
#     # === === ==

#     # Sometimes "season history" will contain
#     # information such as `24F-Taiwan`, or `24-Taiwan`, and in general, one can
#     # expect it to not remain of a fixed format. Therefore, filter out all such
#     # variants apart from the ones we want to explicitly recognize: F*, S*

#     season_history_map = generate_filtered_season_history_map(master_sku_df)

#     master_sku_df = master_sku_df.join(
#         season_history_map, on=["season_history"]
#     )

#     master_sku_df = (
#         master_sku_df.with_columns(
#             sku_year_history=pl.col.season_history_info.list.eval(
#                 pl.element().struct.field("year")
#             )
#             .list.unique()
#             .list.drop_nulls()
#             .list.sort(descending=True)
#         )
#         .with_columns(
#             sku_latest_year=pl.col("sku_year_history")
#             .list.drop_nulls()
#             .list.first()
#         )
#         .filter(pl.col("sku_latest_year").is_not_null())
#     )

#     master_sku_df = master_sku_df.join(
#         master_sku_df.select("category", "sku_year_history")
#         .unique()
#         .group_by("category")
#         .agg(
#             pl.col("sku_year_history")
#             .flatten()
#             .alias("category_year_history"),
#         )
#         .with_columns(
#             pl.col("category_year_history").list.unique().list.sort()
#         ),
#         on="category",
#         how="left",
#         validate="m:1",
#     )
#     # === === ==

#     sku_columns = Sku.members(MemberType.SECONDARY) + ["a_sku", "m_sku"]

#     latest_seasons_df = (
#         (
#             master_sku_df.filter(pl.col.status.eq("active"))
#             .explode("season_history_info")
#             .select("season_history_info", "sku_latest_year", *sku_columns)
#             .filter(
#                 pl.col("season_history_info")
#                 .struct.field("year")
#                 .eq(pl.col("sku_latest_year"))
#                 .or_(
#                     pl.col("season_history_info")
#                     .struct.field("year")
#                     .eq(pl.col("sku_latest_year") - 1)
#                 )
#             )
#             .with_columns(
#                 latest_season=pl.col("season_history_info").struct.field(
#                     "season"
#                 )
#             )
#         )
#         .with_columns(
#             is_fw=pl.col.latest_season.eq("F"),
#             is_ss=pl.col.latest_season.eq("S"),
#         )
#         .group_by(pl.col.category)
#         .agg(
#             pl.col.is_fw.sum().alias("f_tally"),
#             pl.col.is_ss.sum().alias("s_tally"),
#             pl.col.latest_season.alias("latest_seasons"),
#         )
#         .with_columns(total_tally=pl.col.f_tally + pl.col.s_tally)
#         .with_columns(
#             tally_fraction=pl.col.f_tally / pl.col.total_tally,
#             latest_seasons=pl.col.latest_seasons.list.unique()
#             .list.sort()
#             .list.join(","),
#         )
#     ).with_columns(
#         # we manually inspected "tally_fraction", and then determined that the
#         # following categories should be marked FW, even though they
#         # have some PO that happened in SS
#         #
#         # later on, we just read from the CONFIG file to determine seasonality
#         # of items
#         latest_seasons=pl.when(pl.col.category.is_in(["WSF", "WJT", "IHT"]))
#         .then(pl.lit("F"))
#         .otherwise(pl.col.latest_seasons)
#     )

#     season_map = latest_seasons_df.join(
#         latest_seasons_df.select("latest_seasons")
#         .unique()
#         .with_columns(
#             season=pl.col("latest_seasons")
#             .map_elements(Season.map_polars, return_dtype=pl.String())
#             .cast(Season.polars_type())
#         ),
#         on="latest_seasons",
#         how="left",
#         # there are multiple SKU with the same cat in the LHS
#         validate="m:1",
#         nulls_equal=True,
#     ).select("category", "season")

#     master_sku_df = (
#         master_sku_df.join(
#             season_map,
#             on="category",
#             # there are many SKU with the same category in the LHS
#             validate="m:1",
#             nulls_equal=True,
#         )
#         # .drop("season_history_info")
#         .cast(
#             {
#                 "season_history": pl.Enum(
#                     master_sku_df["season_history"].unique().sort()
#                 )
#             }
#         )
#     )

#     master_sku_df = master_sku_df.rename({"m_sku": "sku"})

#     fba_sku = master_sku_df.select(
#         ["a_sku", "sku"]
#         + list(cs.expand_selector(master_sku_df, cs.contains("fba_sku")))
#     )
#     fba_sku = (
#         fba_sku.unpivot(
#             index=["a_sku", "sku"],
#             on=cs.contains("fba_sku"),
#             variable_name="country",
#             value_name="fba_sku",
#         )
#         .drop_nulls()
#         .with_columns(pl.col("country").str.split("fba_sku_").list.last())
#     )
#     fba_sku = fba_sku.cast(
#         {"country": pl.Enum(fba_sku["country"].drop_nulls().unique().sort())}
#     )
#     country_flag = (
#         fba_sku.select("country")
#         .unique()
#         .sort("country")
#         .with_columns(
#             country_flag=pl.col("country").map_elements(
#                 lambda x: CountryFlags.from_str(x).value,
#                 return_dtype=pl.Int64(),
#             )
#         )
#     )
#     fba_sku = fba_sku.join(
#         country_flag, on="country", how="left", validate="m:1", nulls_equal=True
#     ).drop("country")

#     assert (
#         len(
#             master_sku_df.select("print", "print_name")
#             .group_by("print")
#             .agg(pl.col.print_name.unique())
#             .filter(pl.col.print_name.list.len().gt(1))
#         )
#         == 0
#     )

#     print_name_map = (
#         master_sku_df.select("print", "print_name")
#         .unique()
#         .join(
#             convert_dict_to_polars_df(
#                 ADJUSTED_PRINT_NAMES, "print_name", "adjusted_print_name"
#             ).cast({"print_name": master_sku_df["print_name"].dtype}),
#             how="left",
#             on="print_name",
#         )
#         .with_columns(
#             adjusted_print_name=pl.when(pl.col.adjusted_print_name.is_null())
#             .then(pl.col.print_name.cast(pl.String()))
#             .otherwise(pl.col.adjusted_print_name)
#         )
#     )
#     print_name_map = (
#         print_name_map.cast(
#             {
#                 "adjusted_print_name": pl.Enum(
#                     print_name_map["adjusted_print_name"].unique().sort()
#                 )
#             }
#         )
#         .drop("print_name")
#         .rename({"adjusted_print_name": "print_name"})
#     )

#     master_sku_df = extend_df_enum_type(
#         master_sku_df, "size", list(NUMERIC_SIZE.keys())
#     )

#     numeric_size_map = (
#         master_sku_df.select("size")
#         .unique()
#         .join(
#             convert_dict_to_polars_df(
#                 NUMERIC_SIZE, "size", "numeric_size"
#             ).cast(
#                 {
#                     "size": master_sku_df["size"].dtype,
#                     "numeric_size": pl.Float32(),
#                 }
#             ),
#             on=["size"],
#             how="left",
#         )
#         .with_columns(
#             numeric_size=pl.when(pl.col.numeric_size.is_null())
#             .then(pl.col.size.cast(pl.Float32(), strict=False))
#             .otherwise(pl.col.numeric_size)
#         )
#     )

#     problem_sizes = numeric_size_map.filter(pl.col.numeric_size.is_null())
#     if len(problem_sizes) > 0:
#         sys.displayhook(problem_sizes)
#         raise ValueError("Some sizes are not mapped to a numeric size!")

#     master_sku_df = master_sku_df.select(cs.exclude(cs.contains("fba_sku")))
#     result = MasterSkuInfo()
#     result.active_sku = master_sku_df.filter(
#         pl.col("status").eq("active")
#     ).drop("status")
#     result.all_sku = master_sku_df.select(cs.exclude(cs.contains("fba_sku")))
#     result.fba_sku = fba_sku
#     result.ignored_sku = will_ignore
#     result.print_name_map = print_name_map
#     result.numeric_size_map = numeric_size_map

#     return result


# def read_master_sku_excel_file_old(master_sku_date: DateLike) -> MasterSkuInfo:
#     """Read the master information excel file."""
#     master_sku_path = ANALYSIS_INPUT_FOLDER.joinpath(
#         Path(
#             f"1-MasterSKU-All-Product-{Date.from_datelike(master_sku_date).strftime('%Y-%m-%d')}.xlsx"
#         )
#     )

#     assert master_sku_path.is_file(), master_sku_path
#     header_df = pl.read_excel(
#         master_sku_path,
#         sheet_name="MasterFile",
#         read_options={"header_row": MASTER_SKU_DF_HEADER_ROW, "n_rows": 0},
#     ).rename(lambda x: re.sub(r"\s+|_x000D_", " ", x))
#     required_columns = [
#         ix
#         for ix, x in enumerate(header_df.columns)
#         if (
#             (
#                 x.lower()
#                 in [
#                     "msku status",
#                     "seasons sku",
#                     "print sku",
#                     "print name",
#                     "category sku",
#                     "size",
#                     "msku",
#                 ]
#             )
#             or any([y in x.lower() for y in ["fbasku", "adjust", "pause"]])
#         )
#     ]

#     master_sku_df = sanitize_excel_extraction(
#         pl.read_excel(
#             master_sku_path,
#             sheet_name="MasterFile",
#             read_options={
#                 "header_row": MASTER_SKU_DF_HEADER_ROW,
#                 "use_columns": required_columns,
#             },
#         )
#         .rename(lambda x: re.sub(r"\s+|_x000D_", " ", x))
#         .rename(column_renamer)
#         .with_columns(status=pl.col("status").str.to_lowercase())
#     )

#     # we assume that `sku` (aka `m_sku`) is unique
#     assert len(master_sku_df) == len(master_sku_df["m_sku"].unique())

#     master_sku_df = master_sku_df.with_columns(
#         status=pl.col("status").str.to_lowercase().fill_null("NO_STATUS"),
#         season_history=pl.col("season_history").fill_null(""),
#     )
#     # print(master_sku_df["status"].unique().sort())

#     master_sku_df = master_sku_df.filter(
#         pl.col("status").is_in(
#             [
#                 "active",
#                 "inactive",
#                 "retired",
#                 "discontinued",
#                 "cancelled",
#                 "deleted",
#             ]
#         )
#     )

#     master_sku_df = (
#         master_sku_df.with_columns(
#             m_sku_parsed=pl.col("m_sku")
#             .map_elements(
#                 Sku.map_polars,
#                 return_dtype=Sku.intermediate_polars_type_struct(),
#             )
#             .struct.rename_fields(["m_" + k for k in Sku.members()])
#         )
#         .unnest("m_sku_parsed")
#         .with_columns(sku_remainder=pl.col.m_sku_remainder)
#         # .drop("m_print", "m_size")
#         # .rename({"m_sku_remainder": "sku_remainder"})
#         .with_columns(
#             a_sku_parsed=pl.col("a_sku")
#             .map_elements(
#                 Sku.map_polars,
#                 return_dtype=Sku.intermediate_polars_type_struct(),
#             )
#             .struct.rename_fields(["a_" + k for k in Sku.members()])
#         )
#         .unnest("a_sku_parsed")
#         # .drop("a_print", "a_size", "a_sku_remainder")
#     )

#     # === mark special categories ===
#     unique_categories = reduce(
#         lambda x, y: x.extend(y),
#         [master_sku_df[f"{y}category"].unique() for y in ["", "a_", "m_"]],
#     ).unique()
#     is_special_filter = reduce(
#         lambda x, y: x | y,
#         [
#             (
#                 pl.col(v).str.starts_with("M")
#                 & (
#                     pl.col(v).str.slice(1).is_in(unique_categories)
#                     | ~pl.col(v).is_in(STARTS_WITH_M_OK_LIST)
#                 )
#             )
#             | (
#                 pl.col(v).str.ends_with("U")
#                 & (
#                     pl.col(v)
#                     .str.slice(0, length=pl.col(v).str.len_chars() - 1)
#                     .is_in(unique_categories)
#                     | ~pl.col(v).is_in(ENDS_WITH_U_OK_LIST)
#                 )
#             )
#             | pl.col(v).is_in(IGNORE_CATEGORY_LIST)
#             for v in ["category", "a_category", "m_category"]
#         ],
#     )
#     print("Master SKU will ignore:")
#     will_ignore = (
#         master_sku_df.filter(is_special_filter)
#         .select("m_sku", "a_sku", "m_category", "a_category", "category")
#         .rename({"m_sku": "sku"})
#         .unique()
#         .sort("category")
#     )
#     pl.Config.set_tbl_rows(len(will_ignore))
#     sys.displayhook(
#         will_ignore.select(cs.exclude("sku", "a_sku"))
#         .unique()
#         .sort("category")
#     )
#     pl.Config.restore_defaults()
#     master_sku_df = (
#         master_sku_df.filter(~is_special_filter)
#         # master_sku_df.filter(~is_special_filter).drop(
#         #   "a_category", "m_category"
#         # )
#     )
#     # === === ===

#     # === casting columns as enum ===
#     related_columns = {}
#     for z in Sku.members(MemberType.META):
#         related_columns[z] = [
#             f"{y}{z}"
#             for y in ["", "a_", "m_"]
#             if f"{y}{z}" in master_sku_df.columns
#         ]

#     related_casting_columns = []
#     for v in related_columns.values():
#         related_casting_columns += v

#     category_per_related_column = {}
#     for c in related_casting_columns:
#         for z in Sku.members(MemberType.META):
#             if c in related_columns[z]:
#                 category_per_related_column[c] = z

#     other_casting_columns = ["status", "print_name"]

#     default_values_for_related = {}
#     for z in Sku.members(MemberType.META):
#         default_values_for_related[z] = Sku.field_defaults.get(z, None)

#     unique_values_for_related = {}
#     # sometimes, for analysis purposes, we might aggregate data related to a
#     # particular column (e.g. aggregate all "size" sales data into "_ALL_")
#     may_be_combined = [
#         c for c in Sku.members(MemberType.SECONDARY) if c != "category"
#     ]
#     for z in Sku.members(MemberType.META):
#         unique_values = pl.Series(z, [], dtype=pl.String())
#         for c in related_columns[z]:
#             unique_values = unique_values.extend(
#                 master_sku_df[c].unique()
#             ).unique()

#         if z in may_be_combined:
#             unique_values_for_related[z] = (
#                 unique_values.extend(
#                     pl.Series(z, ["_ALL_"], dtype=pl.String())
#                 )
#                 .unique()
#                 .sort()
#             )
#         else:
#             unique_values_for_related[z] = unique_values

#     unique_values = {
#         k: v.sort() for k, v in unique_values_for_related.items()
#     } | {k: master_sku_df[k].unique().sort() for k in other_casting_columns}

#     dtypes: Mapping[str, pl.DataType] = {
#         k: pl.Enum(unique_values[category_per_related_column[k]].sort())
#         for k in related_casting_columns
#         if k in master_sku_df.columns
#     } | {
#         k: pl.Enum(unique_values[k].sort())
#         for k in other_casting_columns
#         if k in master_sku_df.columns
#     }
#     master_sku_df = master_sku_df.cast(dtypes)  # type: ignore
#     # === === ==

#     # === parse pause plans ==
#     unique_plans = pl.Enum(master_sku_df["pause_plan_str"].unique().sort())

#     master_sku_df = (
#         master_sku_df.cast({"pause_plan_str": unique_plans})
#         .join(
#             pl.DataFrame(
#                 [
#                     pl.Series(
#                         "pause_plan_str",
#                         unique_plans.categories,
#                         dtype=unique_plans,
#                     )
#                 ]
#             ).with_columns(
#                 pause_plan=pl.col("pause_plan_str").map_elements(
#                     parse_pause_plan, return_dtype=pl.Int64()
#                 )
#             ),
#             on="pause_plan_str",
#             # we are doing "map_elements" using a join, so we expect m:1
#             validate="m:1",
#             nulls_equal=True,
#         )
#         .drop("pause_plan_str")
#     )
#     # === === ==

#     # Sometimes "season history" will contain
#     # information such as `24F-Taiwan`, or `24-Taiwan`, and in general, one can
#     # expect it to not remain of a fixed format. Therefore, filter out all such
#     # variants apart from the ones we want to explicitly recognize: F*, S*

#     master_sku_df = master_sku_df.with_columns(
#         processed_season_history=pl.col("season_history")
#         .str.split(",")
#         .list.eval(pl.element().str.split(" "))
#         .list.eval(pl.element().explode())
#         .list.eval(pl.element().filter(pl.element().str.len_chars().gt(0)))
#         .list.eval(
#             pl.element()
#             .str.extract_groups(r"(?P<year>\d+)(?P<season>F|S)$")
#             .cast(pl.Struct({"year": pl.Int64(), "season": pl.String()}))
#         )
#         .list.eval(
#             pl.element().filter(
#                 pl.element().struct.field("year").is_not_null()
#                 & pl.element().struct.field("season").is_not_null()
#             )
#         )
#     )

#     master_sku_df = (
#         master_sku_df.with_columns(
#             sku_year_history=pl.col.processed_season_history.list.eval(
#                 pl.element().struct.field("year")
#             )
#             .list.unique()
#             .list.drop_nulls()
#             .list.sort(descending=True)
#         )
#         .with_columns(
#             sku_latest_year=pl.col("sku_year_history")
#             .list.drop_nulls()
#             .list.first()
#         )
#         .filter(pl.col("sku_latest_year").is_not_null())
#     )

#     master_sku_df = master_sku_df.join(
#         master_sku_df.select("category", "sku_year_history")
#         .unique()
#         .group_by("category")
#         .agg(
#             pl.col("sku_year_history")
#             .flatten()
#             .alias("category_year_history"),
#         )
#         .with_columns(
#             pl.col("category_year_history").list.unique().list.sort()
#         ),
#         on="category",
#         how="left",
#         validate="m:1",
#     )
#     # === === ==

#     sku_columns = Sku.members(MemberType.SECONDARY) + ["a_sku", "m_sku"]

#     latest_seasons_df = (
#         (
#             master_sku_df.filter(pl.col.status.eq("active"))
#             .explode("processed_season_history")
#             .select(
#                 "processed_season_history", "sku_latest_year", *sku_columns
#             )
#             .filter(
#                 pl.col("processed_season_history")
#                 .struct.field("year")
#                 .eq(pl.col("sku_latest_year"))
#                 .or_(
#                     pl.col("processed_season_history")
#                     .struct.field("year")
#                     .eq(pl.col("sku_latest_year") - 1)
#                 )
#             )
#             .with_columns(
#                 latest_season=pl.col("processed_season_history").struct.field(
#                     "season"
#                 )
#             )
#         )
#         .with_columns(
#             is_fw=pl.col.latest_season.eq("F"),
#             is_ss=pl.col.latest_season.eq("S"),
#         )
#         .group_by(pl.col.category)
#         .agg(
#             pl.col.is_fw.sum().alias("f_tally"),
#             pl.col.is_ss.sum().alias("s_tally"),
#             pl.col.latest_season.alias("latest_seasons"),
#         )
#         .with_columns(total_tally=pl.col.f_tally + pl.col.s_tally)
#         .with_columns(
#             tally_fraction=pl.col.f_tally / pl.col.total_tally,
#             latest_seasons=pl.col.latest_seasons.list.unique()
#             .list.sort()
#             .list.join(","),
#         )
#     ).with_columns(
#         # we manually inspected "tally_fraction", and then determined that the
#         # following categories should be marked FW, even though they
#         # have some PO that happened in SS
#         #
#         # later on, we just read from the CONFIG file to determine seasonality
#         # of items
#         latest_seasons=pl.when(pl.col.category.is_in(["WSF", "WJT", "IHT"]))
#         .then(pl.lit("F"))
#         .otherwise(pl.col.latest_seasons)
#     )

#     season_map = latest_seasons_df.join(
#         latest_seasons_df.select("latest_seasons")
#         .unique()
#         .with_columns(
#             season=pl.col("latest_seasons")
#             .map_elements(Season.map_polars, return_dtype=pl.String())
#             .cast(Season.polars_type())
#         ),
#         on="latest_seasons",
#         how="left",
#         # there are multiple SKU with the same cat in the LHS
#         validate="m:1",
#         nulls_equal=True,
#     ).select("category", "season")

#     master_sku_df = (
#         master_sku_df.join(
#             season_map,
#             on="category",
#             # there are many SKU with the same category in the LHS
#             validate="m:1",
#             nulls_equal=True,
#         )
#         .drop("processed_season_history")
#         .cast(
#             {
#                 "season_history": pl.Enum(
#                     master_sku_df["season_history"].unique().sort()
#                 )
#             }
#         )
#     )

#     master_sku_df = master_sku_df.rename({"m_sku": "sku"})

#     fba_sku = master_sku_df.select(
#         ["a_sku", "sku"]
#         + list(cs.expand_selector(master_sku_df, cs.contains("fba_sku")))
#     )
#     fba_sku = (
#         fba_sku.unpivot(
#             index=["a_sku", "sku"],
#             on=cs.contains("fba_sku"),
#             variable_name="country",
#             value_name="fba_sku",
#         )
#         .drop_nulls()
#         .with_columns(pl.col("country").str.split("fba_sku_").list.last())
#     )
#     fba_sku = fba_sku.cast(
#         {"country": pl.Enum(fba_sku["country"].drop_nulls().unique().sort())}
#     )
#     country_flag = (
#         fba_sku.select("country")
#         .unique()
#         .sort("country")
#         .with_columns(
#             country_flag=pl.col("country").map_elements(
#                 lambda x: CountryFlags.from_str(x).value,
#                 return_dtype=pl.Int64(),
#             )
#         )
#     )
#     fba_sku = fba_sku.join(
#         country_flag, on="country", how="left", validate="m:1", nulls_equal=True
#     ).drop("country")

#     assert (
#         len(
#             master_sku_df.select("print", "print_name")
#             .group_by("print")
#             .agg(pl.col.print_name.unique())
#             .filter(pl.col.print_name.list.len().gt(1))
#         )
#         == 0
#     )

#     print_name_map = (
#         master_sku_df.select("print", "print_name")
#         .unique()
#         .join(
#             convert_dict_to_polars_df(
#                 ADJUSTED_PRINT_NAMES, "print_name", "adjusted_print_name"
#             ).cast({"print_name": master_sku_df["print_name"].dtype}),
#             how="left",
#             on="print_name",
#         )
#         .with_columns(
#             adjusted_print_name=pl.when(pl.col.adjusted_print_name.is_null())
#             .then(pl.col.print_name.cast(pl.String()))
#             .otherwise(pl.col.adjusted_print_name)
#         )
#     )
#     print_name_map = (
#         print_name_map.cast(
#             {
#                 "adjusted_print_name": pl.Enum(
#                     print_name_map["adjusted_print_name"].unique().sort()
#                 )
#             }
#         )
#         .drop("print_name")
#         .rename({"adjusted_print_name": "print_name"})
#     )

#     numeric_size_map = (
#         master_sku_df.select("size")
#         .unique()
#         .join(
#             convert_dict_to_polars_df(
#                 NUMERIC_SIZE, "size", "numeric_size"
#             ).cast(
#                 {
#                     "size": master_sku_df["size"].dtype,
#                     "numeric_size": pl.Float32(),
#                 }
#             ),
#             on=["size"],
#             how="left",
#         )
#         .with_columns(
#             numeric_size=pl.when(pl.col.numeric_size.is_null())
#             .then(pl.col.size.cast(pl.Float32(), strict=False))
#             .otherwise(pl.col.numeric_size)
#         )
#     )

#     problem_sizes = numeric_size_map.filter(pl.col.numeric_size.is_null())
#     if len(problem_sizes) > 0:
#         sys.displayhook(problem_sizes)
#         raise ValueError("Some sizes are not mapped to a numeric size!")

#     master_sku_df = master_sku_df.select(cs.exclude(cs.contains("fba_sku")))
#     result = MasterSkuInfo()
#     result.active_sku = master_sku_df.filter(
#         pl.col("status").eq("active")
#     ).drop("status")
#     result.all_sku = master_sku_df.select(cs.exclude(cs.contains("fba_sku")))
#     result.fba_sku = fba_sku
#     result.ignored_sku = will_ignore
#     result.print_name_map = print_name_map
#     result.numeric_size_map = numeric_size_map

#     return result
