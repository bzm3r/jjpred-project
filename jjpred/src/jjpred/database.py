"""Generate the primary "database" of historical sales and current inventory
information, usually by reading a ``All Marketplace All SKU Categories``
file."""

from __future__ import annotations

import fastexcel as fxl

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
import os
from typing import Self
from pathlib import Path

import polars as pl
import polars.selectors as cs
from jjpred.analysisdefn import AnalysisDefn, FbaRevDefn
from jjpred.channel import Channel
from jjpred.globalpaths import ANALYSIS_INPUT_FOLDER
from jjpred.globalvariables import IGNORE_CATEGORY_LIST, IGNORE_SKU_LIST

from jjpred.readsupport.instockratio import (
    read_in_stock_ratios_given_meta_info,
)
from jjpred.readsupport.inventory import InventoryType, read_inventory
from jjpred.readsupport.marketing import read_config
from jjpred.readsupport.mastersku import (
    MasterSkuInfo,
    get_master_sku_info,
)
from jjpred.readsupport.utils import (
    cast_standard,
    parse_channels,
    unpivot_dates,
)
from jjpred.sku import Category, Sku
from jjpred.structlike import MemberType, StructLike
from jjpred.utils.datetime import Date

from jjpred.utils.fileio import (
    gen_meta_info_path,
    read_meta_info,
    try_read_df,
    write_df,
)
from jjpred.utils.polars import (
    concat_to_unified,
    struct_filter,
)
from jjpred.utils.typ import as_polars_type, normalize_optional
from jjpred.readsheet import (
    Sheet,
    DataVariant,
    get_relevant_sheets,
)


def gen_primary_excel_data_path(data_date: Date) -> Path:
    """Get the path to the ``All Marketplace All SKU Categories`` file for this
    analysis."""
    possible_paths = [
        ANALYSIS_INPUT_FOLDER.joinpath(
            f"All Marketplace All SKU Categories {data_date.fmt_flat()}.xlsx"
        ),
        ANALYSIS_INPUT_FOLDER.joinpath(
            f"All Marketplace by MSKU Selected SKU Categories {data_date.fmt_flat()}.xlsx"
        ),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise IOError(f"None of possible paths found: {possible_paths=}")


def read_in_stock_ratios(
    analysis_defn_or_database: AnalysisDefn | DataBase,
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
) -> pl.DataFrame:
    """Read in-stock ratios from the ``All Marketplace by MSKU - InStockRatio``
    file."""

    if isinstance(analysis_defn_or_database, DataBase):
        analysis_defn = analysis_defn_or_database.analysis_defn
        active_sku_info = analysis_defn_or_database.meta_info.active_sku
        all_sku_info = analysis_defn_or_database.meta_info.all_sku
    else:
        analysis_defn = analysis_defn_or_database
        active_sku_info = read_meta_info(analysis_defn, "active_sku")
        all_sku_info = read_meta_info(analysis_defn, "all_sku")

    return read_in_stock_ratios_given_meta_info(
        analysis_defn,
        active_sku_info,
        all_sku_info,
        read_from_disk=read_from_disk,
        delete_if_exists=delete_if_exists,
    )


@dataclass
class MetaInfo(MasterSkuInfo):
    """Meta-information useful for conducting analyses."""

    date: pl.DataFrame = field(init=False)
    """The dates associated with the gathered historical information."""
    channel: pl.DataFrame = field(init=False)
    """The channels associated with gathered historical/inventory information."""
    # category_compare: pl.DataFrame = field(init=False)

    @classmethod
    def from_master_sku_result(cls, master_sku_result: MasterSkuInfo) -> Self:
        """Create meta-information based on the SKU information extracted from
        the master sku file."""
        result = cls()
        for f in master_sku_result.fields():
            setattr(result, f, getattr(master_sku_result, f))
        return result

    def update_category_seasons(self, new_season_data: pl.DataFrame):
        """Update the season information associated with this database using
        given season data."""

        assert sorted(new_season_data.columns) == ["category", "season"]

        for f in self.fields():
            df = getattr(self, f)
            if isinstance(df, pl.DataFrame):
                if "season" in df.columns:
                    setattr(
                        self,
                        f,
                        df.join(
                            new_season_data,
                            on=["category"],
                            suffix="_update",
                            how="left",
                        )
                        .with_columns(
                            season=pl.when(pl.col.season_update.is_not_null())
                            .then(pl.col.season_update)
                            .otherwise(pl.col.season)
                        )
                        .drop("season_update"),
                    )


class SkuVariant(Enum):
    """Used to indicate whether a file is indexed by ``adjust_sku`` or
    ``merchant_sku``."""

    A_SKU = auto()
    """Corresponds to ``adjust_sku``."""
    M_SKU = auto()
    """Corresponds to ``merchant_sku``."""


class DataBase:
    """Database of historical sales and current inventory information."""

    analysis_defn: AnalysisDefn
    """Analysis definition information used to initialize this database."""
    dfs: dict[DataVariant, pl.DataFrame]
    """The dataframes containig historical sales/current inventory information."""
    meta_info: MetaInfo
    """Meta-information usedful for conducting analyses."""
    filters: list[StructLike]
    """Optional filters used to limit the channels/SKUs we are interested in."""

    def __init__(
        self,
        analysis_defn: AnalysisDefn,
        filters: list[StructLike] | None = None,
        read_from_disk=True,
    ):
        self.analysis_defn = analysis_defn

        self.variants = [
            variant
            for variant in DataVariant
            if variant not in DataVariant.ignore()
        ]
        self.dfs: dict[DataVariant, pl.DataFrame] = dict()

        master_sku_result = get_master_sku_info(
            self.analysis_defn, read_from_disk=read_from_disk
        )
        self.meta_info = MetaInfo.from_master_sku_result(master_sku_result)

        self.initialize_filters(normalize_optional(filters, []))

        success = False
        if read_from_disk:
            success = self.read_saved_dfs()

        if not success:
            self.generate_from_excel()

        if self.analysis_defn.config_date is not None:
            config_data = read_config(self.analysis_defn)
            self.meta_info.update_category_seasons(config_data.category_season)
            self.save_meta_dfs()

    def dispatch_date(self) -> Date:
        """Get the dispatch date associated with the ID of this database."""
        if isinstance(self.analysis_defn, FbaRevDefn):
            return self.analysis_defn.dispatch_date
        else:
            raise ValueError(
                f"{self.analysis_defn=} which is not of type {type(FbaRevDefn)}"
            )

    @property
    def history(self) -> pl.DataFrame:
        """Get the history dataframe."""
        return self.dfs[DataVariant.History]

    @property
    def in_stock_ratio(self) -> pl.DataFrame:
        """Get the in stock ratio dataframe."""
        return self.dfs[DataVariant.InStockRatio]

    @property
    def inventory(self) -> pl.DataFrame:
        """Get the inventory dataframe."""
        return self.dfs[DataVariant.Inventory]

    def initialize_filters(self, filters: list[StructLike]):
        """Initialize filters based on those that are active, in-season and
        in-focus."""
        assert all(isinstance(x, StructLike) for x in filters)
        self.focus_categories = [
            x.category for x in filters if isinstance(x, Sku)
        ]
        active_categories = self.meta_info.active_sku["category"].unique()
        if len(self.focus_categories) == 0 and len(active_categories) > 0:
            self.focus_categories = list(active_categories)
        elif len(self.focus_categories) > 0 and len(active_categories) > 0:
            self.focus_categories = sorted(
                list(
                    set(self.focus_categories).intersection(active_categories)
                )
            )
        self.filters = list(set(filters))

    def save_meta_dfs(self, overwrite=True) -> dict[str, Path]:
        """Save meta-information dataframes."""
        paths = {}
        for meta_name in self.meta_info.fields():
            write_df(
                overwrite,
                gen_meta_info_path(self.analysis_defn, meta_name),
                getattr(self.meta_info, meta_name),
            )

        return paths

    def save_data_dfs(self, overwrite=False) -> dict[DataVariant, Path]:
        """Same primary information data frames."""
        paths: dict[DataVariant, Path] = DataVariant.gen_save_paths(
            self.analysis_defn
        )
        for variant, df in self.dfs.items():
            if df is not None:
                write_df(
                    overwrite,
                    paths[variant],
                    df,
                )
            else:
                paths.pop(variant)

        return paths

    def generate_from_excel(self):
        """Generate from Excel input files."""
        self.execute_read_from_excel(
            focus_categories=self.focus_categories,
        )
        self.meta_info.date = (
            self.dfs[DataVariant.History]
            .select("date")
            .unique()
            .with_columns(
                pl.col("date").dt.month_end().alias("month_end_date")
            )
            .with_columns(
                (
                    pl.col("month_end_date")
                    - pl.col("date")
                    + pl.duration(days=1)
                )
                .alias("days_in_month")
                .dt.total_days()
                .cast(pl.Int16())
            )
            .drop("month_end_date")
        )
        self.save_all(True)

    def filter(self, additional_filters: Sequence[StructLike] | None = None):
        """Filter this database's information based on its existing filters and
        the additional filters provided."""
        if additional_filters is not None:
            self.initialize_filters(
                list(set(self.filters).union(set(additional_filters)))
            )
        for k, df in self.dfs.items():
            self.dfs[k] = struct_filter(df, *self.filters)

    def save_all(self, overwrite=False) -> dict[DataVariant | str, Path]:
        """Save meta and primary information dataframes."""
        paths = self.save_data_dfs(overwrite)
        paths |= self.save_meta_dfs(overwrite)
        return paths

    def read_saved_dfs(self) -> bool:
        """Read saved dataframes necessary to re-populate this database."""
        success = self.read_data_dfs()
        if not success:
            return success
        success = self.read_meta_dfs()
        return success

    def read_data_dfs(
        self,
    ) -> bool:
        """Read saved data dataframes (in contrast to meta information
        dataframes)."""
        save_paths = DataVariant.gen_save_paths(self.analysis_defn)
        for variant, path in save_paths.items():
            if variant not in self.dfs.keys():
                result = try_read_df(path)
                if result is not None:
                    self.dfs[variant] = result
                else:
                    return False
        self.filter()
        return True

    def execute_read_from_excel(
        self,
        focus_categories: list[Category],
    ):
        """Read historical data and channel inventory data from a ``All
        Marketplace All SKU Categories`` file, and warehouse inventory data from
        a  XORO/Netsuite inventory file."""
        unified_sheets: dict[DataVariant, Sheet | None] = {}

        excel_path = gen_primary_excel_data_path(
            self.analysis_defn.sales_and_inventory_date
        )
        wb = fxl.read_excel(excel_path)
        relevant_sheets = get_relevant_sheets(
            wb,
            [DataVariant.History, DataVariant.Inventory],
            focus_categories,
        )

        for variant, sheet_infos in relevant_sheets.items():
            sheets = list(variant.extract_data(wb, sheet_infos).values())
            if len(sheets) > 0:
                unified_sheet = sheets[0]
                for other in sheets[1:]:
                    unified_sheet.df = pl.concat(
                        [unified_sheet.df, other.df], how="vertical"
                    )
                unified_sheets[variant] = unified_sheet

        for variant, sheet in unified_sheets.items():
            if sheet:
                if variant == DataVariant.History:
                    sheet.df = unpivot_dates(
                        sheet.df, sheet.id_cols, sheet.data_cols, "sales"
                    )
                    # remove multiple entries with all the same identifying
                    # information yet different sales information by adding the
                    # sales
                    sheet.df = sheet.df.group_by(cs.exclude("sales")).agg(
                        pl.col("sales").sum()
                    )
                    # remove any data for SKUs which have 0 sales recorded
                    # over all channels and dates
                    sheet.df = (
                        sheet.df.with_columns(
                            agg_sum=pl.col("sales")
                            .sum()
                            .over(
                                cs.expand_selector(
                                    sheet.df,
                                    cs.exclude("sales", "channel", "date"),
                                )
                            ),
                            is_active=pl.col("a_sku").is_in(
                                self.meta_info.active_sku["a_sku"]
                            ),
                        )
                        .filter(
                            pl.col("agg_sum").gt(0).or_(pl.col("is_active"))
                        )
                        .drop("agg_sum", "is_active")
                    )

                elif sheet.variant == DataVariant.Inventory:
                    sheet.df = sheet.df.melt(
                        id_vars=[str(c) for c in sheet.id_cols],
                        variable_name="channel",
                        value_name="stock",
                    )
                    sheet.df = sheet.df.group_by(cs.exclude("stock")).agg(
                        pl.col("stock").sum()
                    )

        self.dfs = {
            k: v.df for k, v in unified_sheets.items() if v is not None
        }

        ignore_a_skus = (
            self.meta_info.ignored_sku.filter(
                # sometimes, the a_sku for a SKU that should be ignored is
                # a normal SKU...
                pl.col.sku.eq(pl.col.a_sku)
            )["a_sku"]
            .unique()
            .append(pl.Series(IGNORE_SKU_LIST))
        )
        ignore_skus = self.meta_info.ignored_sku["sku"].unique()
        self.dfs[DataVariant.Inventory] = parse_channels(
            cast_standard(
                [self.meta_info.all_sku],
                self.dfs[DataVariant.Inventory].filter(
                    ~pl.col.a_sku.is_in(ignore_a_skus)
                ),
            )
        ).join(self.meta_info.all_sku.select("sku", "a_sku"), on="a_sku")
        self.dfs[DataVariant.Inventory] = self.dfs[
            DataVariant.Inventory
        ].vstack(
            cast_standard(
                [self.meta_info.all_sku],
                read_inventory(
                    self.analysis_defn,
                    InventoryType.AUTO,
                    read_from_disk=False,
                )
                .filter(~pl.col.sku.is_in(ignore_skus))
                .filter(
                    ~pl.col.category.is_in(pl.Series(IGNORE_CATEGORY_LIST))
                ),
            )
            .join(
                self.meta_info.all_sku.select("a_sku", "sku"),
                on="sku",
                how="left",
                validate="m:1",
                join_nulls=True,
            )
            .select(self.dfs[DataVariant.Inventory].columns)
        )

        self.dfs[DataVariant.History] = parse_channels(
            cast_standard(
                [self.meta_info.all_sku],
                self.dfs[DataVariant.History].filter(
                    ~pl.col.a_sku.is_in(ignore_a_skus)
                ),
            ).join(
                # we might want to pick out "category" (corresponding to
                # "category" column in the master SKU file) rather than
                # "a_category" which corresponds only to the category part of
                # a_sku
                self.meta_info.all_sku.select("a_sku", "category").unique(),
                # .rename({"a_category": "category"}),
                on="a_sku",
            )
        )

        # self.dfs = standardize_sku_info(
        #     self.dfs, self.meta_info.all_sku, SkuVariant.A_SKU
        # )

        if self.analysis_defn.in_stock_ratio_date is not None:
            self.dfs[DataVariant.InStockRatio] = read_in_stock_ratios(
                self, delete_if_exists=True
            )

        self.dfs, channel_meta = standardize_channel_info(self.dfs)

        self.meta_info.channel = channel_meta

        self.filter()

    def read_meta_dfs(self) -> bool:
        """Read saved meta information dataframes.

        Returns whether all expected meta information was successfully found."""
        self.meta_info = MetaInfo()
        for meta_name in self.meta_info.fields():
            path = gen_meta_info_path(self.analysis_defn, meta_name)
            df = try_read_df(path)
            if df is not None:
                setattr(self.meta_info, meta_name, df)
            else:
                return False
        return True


def standardize_channel_info(
    dfs: dict[DataVariant, pl.DataFrame],
) -> tuple[dict[DataVariant, pl.DataFrame], pl.DataFrame]:
    """Parse raw string channels in dataframes, interpret them as
    :py:class:`Channel` objects, and cast channel strings as
    :py:class:`polars.Enum`."""

    channels = None
    for df in dfs.values():
        channels = concat_to_unified(
            channels,
            df.select(Channel.members(MemberType.META)).unique(),
        )

    assert channels is not None

    unique_channels = channels.unique()
    unique_channels = unique_channels.cast(
        {"channel": pl.Enum(unique_channels["channel"].unique().sort())}
    )
    # unique_channels = channels.unique().cast(Channel.polars_type_dict())  # type: ignore

    # unique_channels = unique_channels.rename(
    #     {"channel": "raw_channel"}
    # ).with_columns(
    #     channel=pl.struct(Channel.members()).map_elements(
    #         Channel.map_polars_struct_to_string, return_dtype=pl.String()
    #     )
    # )
    # unique_channels = unique_channels.cast(
    #     {"channel": pl.Enum(unique_channels["channel"].unique().sort())}
    # )

    recast_dict = {
        k: unique_channels[k].dtype
        for k in ["channel"]
        if k in unique_channels.columns
    }
    for key, df in dfs.items():
        dfs[key] = df.select(cs.exclude("raw_channel")).cast(recast_dict)  # type: ignore
    # for c in Channel.members():
    #     if unique_channels[c].dtype == pl.String():
    #         pl_enum = pl.Enum(pl.Series(unique_channels[c].unique()))
    #         unique_channels = unique_channels.with_columns(
    #             pl.col(c).cast(pl_enum)
    #         )
    #     elif c == "country_flag":
    #         pl.col(c).cast(PolarsCountryFlagType)

    # for key, df in dfs.items():
    #     if "raw_channel" not in df.columns and "channel" in df.columns:
    #         df = df.rename({"channel": "raw_channel"})
    #     df = df.join(
    #         unique_channels,
    #         on=Channel.members(),
    #         # there are multiple entries with the same channel info
    #         validate="m:m",
    #         join_nulls=True,
    #     )
    #     # df = df.with_columns(
    #     #     pl.col("channel").cast(unique_channels["channel"].dtype)
    #     # ).drop(Channel.members())
    #     # df = df.join(
    #     #     unique_channels.filter(~pl.col("platform").is_null()),
    #     #     on="channel",
    #     #     validate="m:1",
    #     #     join_nulls=True,
    #     # )
    #     dfs[key] = df
    #     # sys.displayhook(df)

    return dfs, unique_channels


def standardize_sku_info(
    dfs: dict[DataVariant, pl.DataFrame],
    all_sku_info: pl.DataFrame,
    sku_variant: SkuVariant,
) -> dict[DataVariant, pl.DataFrame]:
    """Data in inventory files typically comes indexed either by ``adjust_sku``
    (``a_sku``) (e.g. ``All Marketplace All SKU Categories``) or possibly by
    ``merchant_sku`` (``m_sku``).

    In any case, they have to be re-interpreted as :py:class:`Sku` and matched
    against what we know about SKUs from the master SKU information file."""

    if sku_variant == SkuVariant.A_SKU:
        sku_columns = ["a_sku"]
    elif sku_variant == SkuVariant.M_SKU:
        sku_columns = ["sku"]
    else:
        raise ValueError(f"No logic to handle case {sku_variant}!")

    sku_schema = {k: all_sku_info[k].dtype for k in sku_columns}

    for k, df in dfs.items():
        df = df.filter(
            pl.col(x).is_in(as_polars_type(sku_schema[x], pl.Enum).categories)
            for x in sku_columns
        ).cast(sku_schema)  # type: ignore

        df = df.join(
            all_sku_info.select(
                sku_columns + ["category", "a_category"]
            ).unique(),
            on=sku_columns,
            validate="m:1",
            join_nulls=True,
        )

        # with_sku_remainder_info = df.join(
        #     all_sku_info.select(
        #         Sku.members(MemberType.SECONDARY)
        #         + ["a_sku", "sku"]
        #         + [
        #             "pause_plan",
        #             "season",
        #             "sku_year_history",
        #             "category_year_history",
        #             "sku_latest_year",
        #         ]
        #     ).unique(),
        #     on=sku_columns,
        #     validate="m:1",
        #     join_nulls=True,
        # )
        # dfs[k] = with_sku_remainder_info

    return dfs


def standardize_sku_info_old(
    dfs: dict[DataVariant, pl.DataFrame],
    all_sku_info: pl.DataFrame,
    sku_variant: SkuVariant,
) -> dict[DataVariant, pl.DataFrame]:
    """Data in inventory files typically comes indexed either by ``adjust_sku``
    (``a_sku``) (e.g. ``All Marketplace All SKU Categories``) or possibly by
    ``merchant_sku`` (``m_sku``).

    In any case, they have to be re-interpreted as :py:class:`Sku` and matched
    against what we know about SKUs from the master SKU information file."""

    if sku_variant == SkuVariant.A_SKU:
        sku_columns = ["a_sku"]
    elif sku_variant == SkuVariant.M_SKU:
        sku_columns = ["sku"]
    else:
        raise ValueError(f"No logic to handle case {sku_variant}!")

    sku_schema = {k: all_sku_info[k].dtype for k in sku_columns}

    for k, df in dfs.items():
        df = df.filter(
            pl.col(x).is_in(as_polars_type(sku_schema[x], pl.Enum).categories)
            for x in sku_columns
        ).cast(sku_schema)  # type: ignore

        with_sku_remainder_info = df.join(
            all_sku_info.select(
                Sku.members(MemberType.SECONDARY)
                + ["a_sku", "sku"]
                + [
                    "pause_plan",
                    "season",
                    "sku_year_history",
                    "category_year_history",
                    "sku_latest_year",
                ]
            ).unique(),
            on=sku_columns,
            validate="m:1",
            join_nulls=True,
        )
        dfs[k] = with_sku_remainder_info

    return dfs
