"""Utilities for doing file input/output of Polars dataframes or
Excel workbooks."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import logging
from pathlib import Path

import polars as pl
import xlsxwriter as xlw  # type: ignore

from jjpred.analysisdefn import AnalysisDefn
from jjpred.globalpaths import ANALYSIS_OUTPUT_FOLDER, BRIAN_TWK_FOLDER
from jjpred.globalvariables import DEFAULT_STORAGE_FORMAT
from jjpred.utils.datetime import Date, DateLike
from jjpred.utils.excel import convert_df_for_excel


def delete_or_read_df(
    delete_if_exists: bool, save_path: Path
) -> pl.DataFrame | None:
    if delete_if_exists:
        delete_df(save_path)
    else:
        return try_read_df(save_path)

    return None


def read_df(save_path: Path) -> pl.DataFrame:
    print(f"Loading {save_path}...")
    return pl.read_parquet(save_path, memory_map=False)


def try_read_df(save_path: Path, verbose: bool = True) -> pl.DataFrame | None:
    """Try reading a Polars dataframe at the given path. If the read fails,
    return ``None``."""
    try:
        return read_df(save_path)
    except OSError:
        if verbose:
            print(f"Could not find {save_path=}")
        return None


def delete_df(
    save_path: Path, check_if_default_storage_format: bool = True
) -> bool:
    if save_path.exists():
        if (not check_if_default_storage_format) or save_path.name.endswith(
            DEFAULT_STORAGE_FORMAT
        ):
            print(f"Deleting {save_path}...")
            save_path.unlink()
            return True
        else:
            raise ValueError(
                f"{save_path} is not a {DEFAULT_STORAGE_FORMAT} type file."
            )
    else:
        return False


def write_df(overwrite: bool, save_path: Path, df: pl.DataFrame) -> Path:
    """Write given dataframe to the given path."""
    if save_path.exists():
        if overwrite:
            save_path.unlink()
        else:
            raise OSError(f"File already exists at {save_path}.")

    print(f"Saving to {save_path}...")
    df.write_parquet(save_path)
    return save_path


def write_excel(
    save_path: Path, sheet_dict: Mapping[str, pl.DataFrame | None]
) -> Path:
    print(f"Saving to: {save_path}")

    if save_path.exists():
        save_path.unlink()
    with xlw.Workbook(save_path) as workbook:
        for key, df in sheet_dict.items():
            if df is not None:
                convert_df_for_excel(df).write_excel(
                    workbook=workbook, worksheet=key
                )
            # worksheet = workbook.get_worksheet_by_name(key)
            # assert worksheet is not None
            # worksheet.autofit()

    return save_path


def read_storage_file(
    storage_path: Path,
    fallback_info_generator: Callable[[], pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Read storage file at path, and if it does not exist, optionally
    execute the fallback generator in order to instead create the information
    from scratch.
    """
    if not storage_path.exists() and fallback_info_generator is not None:
        return fallback_info_generator()
    return pl.read_parquet(
        storage_path,
        memory_map=False,
    )


def gen_meta_info_path(analysis_defn: AnalysisDefn, meta_name: str) -> Path:
    return ANALYSIS_OUTPUT_FOLDER.joinpath(
        f"{analysis_defn.tag()}_{meta_name}_info.{DEFAULT_STORAGE_FORMAT}"
    )


@dataclass
class Placeholder:
    name: str


def gen_support_info_path_from_tags(
    analysis_defn_tag: str,
    support_name_tag: str,
    support_file_date_tag: str,
    source_name_tag: str,
) -> Path:
    return ANALYSIS_OUTPUT_FOLDER.joinpath(
        f"{str(analysis_defn_tag)}_{support_name_tag}_info"
        + source_name_tag
        + support_file_date_tag
        + f".{DEFAULT_STORAGE_FORMAT}"
    )


def gen_support_info_path(
    analysis_defn: AnalysisDefn,
    support_name: str,
    support_file_date: DateLike | None,
    source_name: str | None = None,
) -> Path:
    if support_file_date is not None:
        support_file_date_tag = (
            f"_{str(Date.from_datelike(support_file_date))}"
        )
    else:
        support_file_date_tag = ""

    if source_name is None:
        source_name_tag = ""
    else:
        source_name_tag = f"_src_{source_name}"

    return gen_support_info_path_from_tags(
        str(analysis_defn),
        support_name,
        source_name_tag,
        support_file_date_tag,
    )

    # return ANALYSIS_OUTPUT_FOLDER.joinpath(
    #     f"{str(analysis_defn)}_{support_name}_info"
    #     + source_part
    #     + date_part
    #     + f".{DEFAULT_STORAGE_FORMAT}"
    # )


def gen_isr_year_info_path(years: int | list[int] | None) -> Path:
    if years is None:
        year_info = ""
    else:
        if not isinstance(years, list):
            years = [years]
        year_info = "_" + "-".join(str(x) for x in years)

    return BRIAN_TWK_FOLDER.joinpath("InStockRatioData").joinpath(
        f"isr{year_info}.parquet"
    )


def read_meta_info(
    analysis_defn: AnalysisDefn,
    meta_name: str,
    info_generator: Callable[[], pl.DataFrame] | None = None,
) -> pl.DataFrame:
    return read_storage_file(
        gen_meta_info_path(analysis_defn, meta_name), info_generator
    )


def disable_fastexcel_dtypes_logger():
    logging.getLogger("fastexcel.types.dtype").setLevel(logging.ERROR)
