"""Tools for reading from an Excel data sheet containing historical sales or
channel inventory information."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Self, TypeGuard
import fastexcel as fxl
import polars as pl
from jjpred.analysisdefn import AnalysisDefn
from jjpred.columns import (
    IN_STOCK_RATIO_SHEET_COLUMN_DEFNS,
    Column,
    ColumnDefns,
    HISTORY_SHEET_COLUMN_DEFNS,
    INVENTORY_SHEET_COLUMN_DEFNS,
)
import polars.selectors as cs
from jjpred.globalpaths import ANALYSIS_OUTPUT_FOLDER
from jjpred.globalvariables import DEFAULT_STORAGE_FORMAT
from jjpred.sku import Category
import re

from jjpred.utils.polars import sanitize_excel_extraction
from jjpred.utils.str import indent
from jjpred.utils.typ import ScalarOrList, normalize_as_list

SHEET_NAME_RE_PATTERN = re.compile(r"(?P<category>[^-]+)(-(?P<suffix>.*))?")


def get_relevant_sheets(
    wb: fxl.ExcelReader,
    required_variants: ScalarOrList[DataVariant],
    focus_categories: list[Category],
) -> dict[DataVariant, list[SheetInfo]]:
    """Figure out which particular sheets need to be read, based on which
    categories we want to focus on."""
    sheet_infos = get_sheet_info(wb)
    required_variants = normalize_as_list(required_variants)

    relevant_sheets: dict[DataVariant, list[SheetInfo]] = {
        k: [] for k in DataVariant if k in required_variants
    }
    for sheet_info in sheet_infos:
        if matches := re.match(SHEET_NAME_RE_PATTERN, sheet_info.name):
            if (
                len(focus_categories) > 0
                and matches["category"] in focus_categories
            ) or len(focus_categories) == 0:
                variant = DataVariant.from_suffix(matches["suffix"])
                if variant is not None and variant in required_variants:
                    relevant_sheets[variant].append(sheet_info)

    return relevant_sheets


class DataVariantMeta(NamedTuple):
    """Meta-information associated with a particular sheet type."""

    suffix: str
    """Sheet name suffix.

    For example: for HCF0-SKU the ``suffix`` is SKU. For HCF0-SKU-WK the
    ``suffix`` is SKU-WK."""
    col_defns: ColumnDefns = ColumnDefns([], [])
    """Defines the ID vs DATA columns in a sheet."""
    header_range: tuple[int, int] = (0, 0)
    """The rows which should be combined to create header labels"""
    min_row: int = 0
    max_row: Callable[[int], int] = lambda _: 0
    skip_intermediate_columns: list[re.Pattern[str]] = []
    """Some intermediate columns should be skipped if they match."""


@dataclass
class SheetInfo:
    """Information characterizing a sheet: its name, and the number of rows it
    contains."""

    name: str
    num_rows: int


def get_sheet_info(wb: fxl.ExcelReader) -> list[SheetInfo]:
    sheet_names = wb.sheet_names
    sheet_heights = [
        # by default,
        wb.load_sheet(name, header_row=None).total_height
        for name in sheet_names
    ]
    return list(
        SheetInfo(name, height)
        for name, height in zip(sheet_names, sheet_heights)
    )


def fill_unnamed_cols(
    header_df: pl.DataFrame,
    replace_whitespace: str | None = None,
) -> dict[str, str]:
    """Fill unnamed columns based on the last encountered column label, and
    information in the rows below.

    Arguments:
        header_df (pl.DataFrame): a small Polars DataFrame with columns
        corresponding to the original columns, and rows that contain extra
        column label information.
        replace_whitespace (str | None): replace whitespace in
        column names with the given string.
    """
    names = []
    last_name = None
    for oc in header_df.columns:
        if "unnamed" in oc.lower():
            names.append(last_name)
        else:
            names.append(oc)
            last_name = oc

    if replace_whitespace is not None:
        names = [re.sub(r"\s+", "_", x) for x in names]

    original_to_renamed = dict(
        (
            k,
            str(
                " ".join(
                    [v]
                    + list(
                        (str(x) if x is not None else "") for x in header_df[k]
                    )
                )
            ).strip(),
        )
        for k, v in zip(header_df.columns, names)
    )

    return original_to_renamed


@dataclass
class Sheet:
    """A Polars representation of an Excel data sheet containing historical
    sales or channel inventory information."""

    df: pl.DataFrame
    category: Category
    variant: DataVariant
    name: str
    id_cols: list[Column]
    data_cols: list[Column]


class DataVariant(DataVariantMeta, Enum):
    History = (
        "SKU",
        HISTORY_SHEET_COLUMN_DEFNS,
        (0, 2),
        2,
        lambda max_row: max_row - 3,  # we want to ignore summary rows,
        [
            re.compile(r"^Per Sales Channel$"),
            re.compile(r"^\d{4} Sales Quantity$"),
        ],
    )
    """A sheet containing historical sales information."""
    Inventory = (
        "SKU-WK",
        INVENTORY_SHEET_COLUMN_DEFNS,
        (1, 2),
        2,
        lambda max_row: max_row,
        [],
    )
    """A sheet containing channel inventory information."""
    InStockRatio = (
        "InStockR-M",
        IN_STOCK_RATIO_SHEET_COLUMN_DEFNS,
        (0, 2),
        2,
        lambda max_row: max_row,
        [
            re.compile(r"^Per Warehouse Location$"),
            re.compile(r"^\d{4}$"),
        ],
    )
    Totals = ""
    """A sheet containing total sales information. (Ignored.)"""

    @classmethod
    def required(cls) -> list[DataVariant]:
        """Get a list of required data variants (i.e. not to be ignored.)"""
        return [
            x
            for x in cls
            if x in [cls.History, cls.Inventory, cls.InStockRatio]
        ]

    @classmethod
    def ignore(cls) -> list[DataVariant]:
        """Get a list of data variants that are to be ignored."""
        return [x for x in cls if x not in cls.required()]

    @classmethod
    def from_suffix(cls, s: str | None) -> Self | None:
        """Determine what type of sheet this is based on its suffix (e.g.
        ``SKU-WK`` is inventory information, while ``SKU`` alone is historical
        sales.)."""
        if not s:
            s = ""

        for case in cls:
            if case.suffix == s:
                return case

    def rename_raw_headers(
        self, original_to_intermediate: dict[str, str], header_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Rename the original columns of the Excel sheet to something
        informative."""
        dropcols = {
            oc for oc, nc in original_to_intermediate.items() if nc is None
        }
        header_df = (
            header_df.drop(dropcols)
            .rename(original_to_intermediate)
            .with_row_index()
            .filter(pl.col("index").ge(0))
            .drop("index")
        )

        return header_df

    def get_relevant_columns(
        self, header_df: pl.DataFrame
    ) -> tuple[
        list[Column],
        list[Column],
        dict[str, str],
        dict[str, type[pl.DataType]],
    ]:
        """Get the columns of the Excel sheet that are relevant to us."""

        null_columns = []
        for c in header_df.columns:
            if header_df[c].dtype == pl.Null() and "UNNAMED" in c:
                null_columns.append(c)
        if len(null_columns) > 0:
            header_df = header_df.select(cs.exclude(null_columns))

        original_to_intermediate = dict(
            [
                (k, v)
                for k, v in fill_unnamed_cols(header_df).items()
                if all(
                    x.match(v) is None for x in self.skip_intermediate_columns
                )
            ]
        )
        header_df = self.rename_raw_headers(
            original_to_intermediate, header_df
        )
        id_cols: list[Column] = []
        data_cols: list[Column] = []

        for index, parts in enumerate(header_df.columns):
            for defn in self.col_defns.id_cols:
                if component := defn.match_raw(parts, index):
                    id_cols.append(component)
                    break
            else:
                for defn in self.col_defns.data_cols:
                    if component := defn.match_raw(parts, index):
                        data_cols.append(component)
                        break

        final_cols = [c for c in id_cols + data_cols]
        wanted_ixs = [c.index for c in final_cols]
        wanted_ocs = [
            oc for ix, oc in enumerate(header_df.columns) if ix in wanted_ixs
        ]

        intermediate_to_final = {
            oc: c.name for oc, c in zip(wanted_ocs, final_cols)
        }
        # intermediate_to_final_dtype = {
        #     oc: c.dtype for oc, c in zip(wanted_ocs, final_components)
        # }
        filtered_original_to_intermediate = {
            oc: ic
            for oc, ic in original_to_intermediate.items()
            if ic is not None and ic in intermediate_to_final.keys()
        }
        return (
            id_cols,
            data_cols,
            {
                oc: intermediate_to_final[ic]
                for oc, ic in filtered_original_to_intermediate.items()
            },
            {c.name: c.dtype for c in final_cols},
        )

    def extract_data(
        self, wb: fxl.ExcelReader, sheet_infos: list[SheetInfo]
    ) -> dict[str, Sheet]:
        """Extract data from an Excel workbook into mapping with keys
        corresponding to the name of the sheet in the workbook and
        data corresponding to the list of extracted sheets associated with that
        type of data."""
        first_header_row = self.header_range[0]
        num_header_rows = self.header_range[-1] - self.header_range[0]

        result: dict[str, Sheet] = {}
        for sheet_info in sheet_infos:
            print(f"Extracting data from {sheet_info.name}...")
            header_df = wb.load_sheet(
                sheet_info.name,
                header_row=first_header_row,
                n_rows=num_header_rows - 1,
            ).to_polars()

            id_cols, data_cols, column_rename_map, schema = (
                self.get_relevant_columns(header_df)
            )

            sheet_rows = self.max_row(sheet_info.num_rows)
            if sheet_rows > num_header_rows:
                n_rows = sheet_rows - num_header_rows
                skip_rows = num_header_rows - 1
                if skip_rows < 0:
                    skip_rows = 0
            else:
                n_rows = 0
                skip_rows = 0

            # n_rows = self.max_row(sheet_info.num_rows) - skip_rows
            # if n_rows < 0:
            #     n_rows = 0

            df = (
                wb.load_sheet(
                    sheet_info.name,
                    header_row=first_header_row,
                    skip_rows=skip_rows,
                    n_rows=n_rows,
                    use_columns=list(column_rename_map.keys()),
                )
                .to_polars()
                .rename(column_rename_map)
                .cast(schema)  # type: ignore
            )

            final_df = sanitize_excel_extraction(
                df.select(cs.exclude(cs.numeric())).with_columns(
                    df.select(cs.numeric()).fill_null(strategy="zero")
                )
            )

            category = ""
            if match := SHEET_NAME_RE_PATTERN.match(sheet_info.name):
                category = match.groupdict()["category"]
            else:
                raise Exception(
                    f"sheet {sheet_info.name} has no category match!"
                )
            if final_df.shape[0] > 0:
                result[sheet_info.name] = Sheet(
                    final_df,
                    category,
                    self,
                    f"{category} {self.name}",
                    id_cols,
                    data_cols,
                )
                print(indent("Done."))
            else:
                print(indent(f"Skipping {sheet_info.name}: no data.", 4))

        return result

    def gen_save_path(self, analysis_defn: AnalysisDefn) -> Path:
        """Generate the path where extracted information for this data variant
        will be saved. (This can also be used to figure out where to read this
        saved data from later.)"""
        return ANALYSIS_OUTPUT_FOLDER.joinpath(
            f"{analysis_defn.tag()}_{self.name}.{DEFAULT_STORAGE_FORMAT}"
        )

    @classmethod
    def gen_save_paths(
        cls, analysis_defn: AnalysisDefn
    ) -> dict[DataVariant, Path]:
        """Generate save paths per data variant. (see also ``gen_save_path``.)"""
        result = []
        for variant in cls.required():
            result.append(
                (
                    variant,
                    variant.gen_save_path(analysis_defn),
                )
            )
        return dict(result)

    def lower(self) -> str:
        """Get the name of this data variant in lowercase form."""
        return self.name.lower()

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return (f"{self.name}").__hash__()

    def _is_valid_operand(self, other) -> TypeGuard[Self]:
        return isinstance(other, self.__class__) or isinstance(other, str)

    def __lt__(self, other: object) -> bool:
        if self._is_valid_operand(other):
            if isinstance(other, self.__class__):
                return self.name < other.name
            elif isinstance(other, str):
                return self.name < other

        raise Exception(f"Cannot compare {self.__class__} with {type(other)}")

    def __eq__(self, other: object) -> bool:
        if self._is_valid_operand(other):
            if isinstance(other, self.__class__):
                return self.name == other.name
            elif isinstance(other, str):
                return self.name == other

        raise Exception(f"Cannot compare {self.__class__} with {type(other)}")
