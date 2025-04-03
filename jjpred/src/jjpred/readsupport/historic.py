"""Functions to read from the :code:`Historic sales and inv` file."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from jjpred.analysisdefn import FbaRevDefn
from jjpred.channel import Channel
from jjpred.globalpaths import ANALYSIS_INPUT_FOLDER
from jjpred.readsupport.utils import (
    cast_standard,
)
from jjpred.utils.datetime import Date, DateLike
from jjpred.utils.fileio import (
    delete_or_read_df,
    gen_support_info_path,
    read_meta_info,
    write_df,
)
from jjpred.utils.multidict import MultiDict
from jjpred.utils.polars import (
    polars_float,
    vstack_to_unified,
    sanitize_excel_extraction,
    struct_filter,
)
from jjpred.utils.typ import as_polars_type, normalize_optional


HISTORIC_SALES_AND_INV_FILE: str = (
    "Historic sales and inv. data for all cats v{version} ({date}).xlsx"
)
"""Format string for the name of the ``Historic sales and inv`` file."""


def gen_historic_sales_and_inv_path(
    file_date: DateLike, start: int = 57, max_tries: int = 100
) -> Path:
    """Find the highest version ``Historic sales and inv`` data for all cats with the given date."""
    file_date = Date.from_datelike(file_date)
    for version in reversed(range(start, start + max_tries)):
        calculation_path = ANALYSIS_INPUT_FOLDER.joinpath(
            HISTORIC_SALES_AND_INV_FILE.format(
                version=version, date=file_date.fmt_flat()
            )
        )
        if calculation_path.exists():
            print(f"{calculation_path} exists!")
            return calculation_path

    path_shape = HISTORIC_SALES_AND_INV_FILE.format(
        version="{NUMBER}", date=file_date.fmt_flat()
    )
    raise OSError(
        f"Could not find valid calculation file for {str(file_date)}. "
        f"Should have shape: {path_shape}"
    )


def read_mon_sale_r_params(
    analysis_defn: FbaRevDefn,
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
    overwrite: bool = True,
):
    """Read calculation parameter information (e.g. historical year, reference channel, reference category, etc.) from the ``Historic sales and inv`` data for all cats calculation file."""

    mon_sale_r_date = analysis_defn.get_mon_sale_r_date()
    save_path = gen_support_info_path(
        analysis_defn,
        "mon_sale_r",
        mon_sale_r_date,
        source_name="historicfile",
    )
    if read_from_disk or delete_if_exists:
        result = delete_or_read_df(delete_if_exists, save_path)
        if result is not None:
            return result

    active_sku_info = read_meta_info(analysis_defn, "active_sku")
    channel_info = read_meta_info(analysis_defn, "channel")
    calculation_path = gen_historic_sales_and_inv_path(mon_sale_r_date)

    calc_df = sanitize_excel_extraction(
        pl.read_excel(calculation_path, sheet_name="MonSaleR")
    ).rename({"HELP": "category"})

    channels = [
        x
        for x in enumerate(calc_df.columns)
        if (x[1] not in ["category"]) and ("unnamed" not in x[1].lower())
    ]
    calc_df = (
        calc_df.reverse()
        .with_columns(pl.col("category").forward_fill(1))
        .with_row_index()
    )
    calc_info_df = calc_df.filter(pl.col("index").mod(2).eq(1)).drop("index")

    renames = {
        "category": 0,  # column 0
        "historical_year": 4,  # column 4
        "latest_historical_month": 6,  # column 6
        "ref_cat": 8,  # column 8
        "ref_channel": 10,  # column 10
    }

    calc_params_df = pl.DataFrame()
    for ix, channel in channels:
        channel_lower = channel.lower()

        if any(
            [
                x in channel_lower
                for x in [
                    "all channels",
                    "working year",
                    "completed month",
                ]
            ]
        ):
            continue

        # skip Working Year value
        if len(channel_lower) > 0:
            try:
                _ = int(channel_lower)
                continue
            except ValueError:
                pass

        ch = Channel.parse(channel)
        this_df = calc_info_df.select("category", pl.nth(range(ix, ix + 12)))
        rename_map = dict(
            zip(
                [
                    c
                    for ix, c in enumerate(this_df.columns)
                    if ix in renames.values()
                ],
                renames.keys(),
                strict=True,
            )
        )
        this_df = (
            this_df.rename(rename_map)
            .select(renames)
            .with_columns(**ch.to_columns(remove_defaults=False))
            .cast(
                {
                    "historical_year": pl.UInt16(),
                    "latest_historical_month": pl.Int8(),
                }
            )
        )
        calc_params_df = vstack_to_unified(
            calc_params_df,
            this_df,
        )

    missing_from_active_sku = [
        x
        for x in calc_params_df["category"].unique()
        if x not in active_sku_info["category"].unique()
    ]
    print(f"Historic-file read params will ignore: {missing_from_active_sku=}")
    print(f"({len(missing_from_active_sku)=})")
    calc_params_df = cast_standard(
        [active_sku_info],
        calc_params_df.filter(~pl.col.category.is_in(missing_from_active_sku)),
        {
            "ref_cat": "category",
        },
    )

    write_df(overwrite, save_path, calc_params_df)

    return calc_params_df


def read_demand_ratios(
    analysis_defn: FbaRevDefn,
    read_from_disk: bool = True,
    delete_if_exists=False,
    overwrite: bool = True,
):
    """Read demand ratios (monthly ratios) from ``Historic sales and inv`` data for all cats file."""
    save_path = gen_support_info_path(
        analysis_defn,
        "demand_ratio",
        analysis_defn.get_mon_sale_r_date(),
        source_name="historicfile",
    )

    if read_from_disk or delete_if_exists:
        result = delete_or_read_df(delete_if_exists, save_path)
        if result is not None:
            return result

    active_sku_info = read_meta_info(analysis_defn, "active_sku")
    channel_info = read_meta_info(analysis_defn, "channel")
    calculation_path = gen_historic_sales_and_inv_path(
        analysis_defn.get_mon_sale_r_date()
    )

    calc_df = sanitize_excel_extraction(
        pl.read_excel(calculation_path, sheet_name="MonSaleR")
    ).rename({"HELP": "category"})

    channels = [
        x
        for x in enumerate(calc_df.columns)
        if (x[1] not in ["category"]) and ("unnamed" not in x[1].lower())
    ]
    calc_df = (
        calc_df.reverse()
        .with_columns(pl.col("category").forward_fill(1))
        .with_row_index()
    )
    calc_info_df = calc_df.filter(pl.col("index").mod(2).eq(0)).drop("index")

    renames = [
        "category",  # column 0
    ] + [str(x) for x in range(1, 13)]
    monthly_demand_ratio_df = pl.DataFrame()
    for ix, channel in channels:
        channel_lower = channel.lower()
        if any(
            [
                x in channel_lower
                for x in [
                    "all channels",
                    "working year",
                    "completed month",
                ]
            ]
        ):
            continue

        # skip Working Year value
        if len(channel_lower) > 0:
            try:
                _ = int(channel_lower)
                continue
            except ValueError:
                pass

        ch = Channel.parse(channel)
        this_df = calc_info_df.select(
            "category", pl.nth(range(ix, ix + 12))
        ).filter(
            pl.col("category").is_in(
                as_polars_type(
                    active_sku_info["category"].dtype, pl.Enum
                ).categories
            )
        )
        rename_map = dict(
            zip(
                this_df.columns,
                renames,
                strict=True,
            )
        )
        this_df = (
            this_df.rename(rename_map)
            .select(renames)
            .unpivot(index="category")
            .rename({"variable": "month", "value": "demand_ratio"})
            .with_columns(
                pl.col("demand_ratio").replace(
                    {k: np.nan for k in ["ERROR", "ERROR OPTIONS"]}
                )
            )
            .cast({"month": pl.Int8(), "demand_ratio": polars_float(64)})
            .with_columns(**ch.to_columns(remove_defaults=False))
        )
        monthly_demand_ratio_df = vstack_to_unified(
            monthly_demand_ratio_df,
            this_df,
        )

    monthly_demand_ratio_df = monthly_demand_ratio_df.filter(
        ~pl.col.platform.eq("Wholesale")
    )
    monthly_demand_ratio_df = cast_standard(
        [active_sku_info, channel_info], monthly_demand_ratio_df
    )
    write_df(overwrite, save_path, monthly_demand_ratio_df)

    return monthly_demand_ratio_df


class CalcParamMeta(NamedTuple):
    """Information stored with each calculation parameter enum variant."""

    column_name: str
    null_replace_value: str | int | None = None


class CalcParamType(CalcParamMeta, Enum):
    """:py:class:`Enum` enumerating the parameters available in ``Historic sales and inv`` data for all cats file."""

    ReferenceCategories = "ref_cat", None
    """Some categories use reference categories for historical data and therefore monthly ratios."""
    ReferenceChannels = "ref_channel", None
    """Some categories use reference channels for aggregating historical data in order to calculate monthly ratios."""
    HistoricalYear = "historical_year", None
    """Some categories don't use ``2023`` as the default historical year."""
    LatestHistoricalMonth = "latest_historical_month", None
    """Some categories don't use ``December`` (``12``) as the end of the historical year."""

    def filter(
        self,
        calc_file_df: pl.DataFrame,
        remove_missing: bool = True,
        additional_filters: list[pl.Expr] | None = None,
    ) -> pl.DataFrame:
        """Filter the calculation file dataframe for this particular parameter's
        information."""
        intermediate = calc_file_df.select(
            "category", self.column_name, *Channel.members()
        )

        if remove_missing:
            if self.null_replace_value is None:
                filter_expr = pl.col(self.column_name).is_not_null()
            else:
                filter_expr = pl.col(self.column_name).ne(
                    self.null_replace_value
                )
            intermediate = intermediate.filter(filter_expr)

        additional_filters = normalize_optional(additional_filters, [])
        if len(additional_filters) > 0:
            intermediate = intermediate.filter(*additional_filters)

        return intermediate


def read_specific_calc_param_multidict(
    analysis_defn: FbaRevDefn,
    specific_calc_param: CalcParamType,
    additional_filters: list[pl.Expr] | None = None,
    remove_missing: bool = False,
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
    overwrite: bool = True,
) -> dict[Channel, MultiDict[str, str]]:
    """Read a specific calculation parameter from the ``Historic sales and inv`` data for all cats file, returning a multi-valued (tuple-valued) dictionary as output."""
    mon_sale_r = read_mon_sale_r_params(
        analysis_defn,
        read_from_disk=read_from_disk,
        delete_if_exists=delete_if_exists,
        overwrite=overwrite,
    )
    result = {}
    specific_param_df = specific_calc_param.filter(
        mon_sale_r,
        additional_filters=additional_filters,
        remove_missing=remove_missing,
    )
    for channel in mon_sale_r.select(channel=pl.struct(Channel.members()))[
        "channel"
    ].unique():
        assert isinstance(channel, dict)
        ch = Channel.from_dict(channel)
        if ch not in result.keys():
            result[ch] = MultiDict.from_dict(
                dict(
                    struct_filter(
                        specific_param_df,
                        ch,
                    )
                    .select("category", specific_calc_param.column_name)
                    .rows()
                )
            )
        else:
            raise ValueError(f"Channel {ch} already in result: {result}.")

    return result


def read_specific_calc_param(
    analysis_defn: FbaRevDefn,
    specific_calc_param: CalcParamType,
    additional_filters: list[pl.Expr] | None = None,
    remove_missing: bool = False,
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
    overwrite: bool = True,
) -> pl.DataFrame:
    """Read a specific calculation parameter from the ``Historic sales and inv`` data for all cats file, returning a dataframe as output."""
    return specific_calc_param.filter(
        read_mon_sale_r_params(
            analysis_defn,
            read_from_disk=read_from_disk,
            delete_if_exists=delete_if_exists,
            overwrite=overwrite,
        ),
        additional_filters=additional_filters,
        remove_missing=remove_missing,
    )
