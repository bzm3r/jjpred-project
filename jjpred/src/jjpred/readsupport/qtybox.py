from dataclasses import dataclass, field
import sys
import polars as pl
from pathlib import Path

from jjpred.analysisdefn import FbaRevDefn
from jjpred.globalpaths import ANALYSIS_INPUT_FOLDER
from jjpred.readsupport.utils import cast_standard
from jjpred.utils.fileio import (
    delete_or_read_df,
    gen_support_info_path,
    read_meta_info,
    write_df,
)

from jjpred.utils.datetime import Date, DateLike
from jjpred.utils.polars import find_dupes, sanitize_excel_extraction
from jjpred.utils.typ import as_list

QTY_BOX_FILE: str = "PO boxes and volume - All seasons{date_part}.xlsx"


def gen_qty_box_path(file_date: DateLike | None) -> Path:
    """Get the path of the main program file (``PO boxes and volume - All
    seasons ({date_part}).xlsx``) with the matching date.

    The date is optional and can be omitted.

    If the file is not found, raise an error."""

    if file_date is None:
        date_part = ""
    else:
        date_part = f" ({str(Date.from_datelike(file_date))})"

    path = ANALYSIS_INPUT_FOLDER.joinpath(
        QTY_BOX_FILE.format(date_part=date_part)
    )
    if path.exists():
        print(f"{path} exists!")
        return path

    # path_shape = QTY_BOX_FILE.format(date_part=" (OPTIONAL_DATE)")
    raise OSError(f"Could not find valid qty box file: {path}")


@dataclass
class SheetInfo:
    use_columns: list[int] = field(default_factory=list)
    raw_names: list[str] = field(default_factory=list)
    required_names: list[str] = field(default_factory=list)
    data_types: list[pl.DataType] = field(default_factory=list)

    def append(
        self,
        use_column: int,
        raw_name: str,
        required_name: str,
        data_type: pl.DataType,
        overwrite: bool = False,
    ):
        if required_name in self.required_names:
            if not overwrite:
                raise ValueError(
                    f"{required_name=} already in {self.required_names=}"
                )
            else:
                index = self.required_names.index(required_name)
                self.use_columns[index] = use_column
                self.raw_names[index] = raw_name
                assert self.data_types[index] == data_type
        else:
            assert use_column not in self.use_columns
            assert raw_name not in self.raw_names

            self.required_names.append(required_name)
            self.use_columns.append(use_column)
            self.raw_names.append(raw_name)
            self.data_types.append(data_type)

    def rename_map(self) -> dict[str, str]:
        return dict(
            (k, v)
            for k, v in zip(self.raw_names, self.required_names, strict=True)
        )

    def schema(self) -> dict[str, pl.DataType]:
        return dict(
            (k, v)
            for k, v in zip(self.raw_names, self.data_types, strict=True)
        )


@dataclass
class InfoPerSheet:
    meta_infos: dict[str, SheetInfo] = field(default_factory=dict)

    def append(
        self,
        sheet_name: str,
        use_column: int,
        raw_name: str,
        required_name: str,
        data_type: pl.DataType,
        overwrite: bool = False,
    ):
        if sheet_name not in self.meta_infos.keys():
            self.meta_infos[sheet_name] = SheetInfo()

        self.meta_infos[sheet_name].append(
            use_column, raw_name, required_name, data_type, overwrite
        )


def read_qty_box(
    analysis_defn: FbaRevDefn,
    read_from_disk: bool = True,
    delete_if_exists: bool = False,
):
    """Read qty/box information from the main program file."""

    save_path = gen_support_info_path(
        analysis_defn,
        "qty_box",
        analysis_defn.qty_box_date,
        source_name="po_boxes",
    )

    if read_from_disk:
        result = delete_or_read_df(delete_if_exists, save_path)
        if result is not None:
            return result

    qty_box_path = gen_qty_box_path(analysis_defn.qty_box_date)
    all_sku_info = read_meta_info(analysis_defn, "all_sku")
    # channel_info = read_meta_info(analysis_defn, "channel")

    required_sheets = ["SS", "FW"]

    sheet_headers: dict[str, pl.DataFrame] = pl.read_excel(
        qty_box_path,
        sheet_name=required_sheets,  # type: ignore
        read_options={
            "header_row": 0,
            "n_rows": 0,
        },
        raise_if_empty=False,
    )

    info_per_sheet = InfoPerSheet()
    search_cols = {
        "category": ("cat", pl.String()),
        "size": ("size", pl.String()),
        "qty_box": (["qty", "box"], pl.Int64()),
    }
    for sheet_name, headers in sheet_headers.items():
        for ix, col in enumerate(headers.columns):
            for required_name, search_info in search_cols.items():
                search_item, data_type = search_info
                if all([y in col.lower() for y in as_list(search_item)]):
                    info_per_sheet.append(
                        sheet_name,
                        ix,
                        col,
                        required_name,
                        data_type,
                        overwrite=("category" == required_name),
                    )

    sheet_dfs = []
    for sheet_name, sheet_info in info_per_sheet.meta_infos.items():
        sheet_dfs.append(
            cast_standard(
                [all_sku_info],
                sanitize_excel_extraction(
                    pl.read_excel(
                        qty_box_path,
                        sheet_name=sheet_name,
                        read_options={
                            "header_row": 0,
                            "use_columns": sheet_info.use_columns,
                        },
                        schema_overrides=sheet_info.schema(),
                    ).rename(sheet_info.rename_map())
                ),
            ).filter(~(pl.col.qty_box.is_null() | pl.col.qty_box.eq(0)))
        )
    qty_box_df = pl.concat(sheet_dfs)

    # Check if any dupes have different `qty_box`` based on season.
    # We are currently assuming that the `qty_box`` size is the same regardless
    # of dispatch season!
    dupes = find_dupes(qty_box_df, ["category", "size"])
    if len(dupes) > 0:
        dupes = dupes.with_columns(
            uniques=pl.col.qty_box.list.unique()
        ).with_columns(is_unique=pl.col.uniques.list.len().eq(1))
        if len(dupes.filter(~pl.col.is_unique)) > 0:
            sys.displayhook(dupes.filter(~pl.col.is_unique))
            raise ValueError("Found SKUs with multiple qty_box entries!")
    qty_box_df = qty_box_df.unique()

    qty_box_info = all_sku_info.join(
        qty_box_df, on=["category", "size"]
    ).select("sku", "a_sku", "qty_box")

    write_df(True, save_path, qty_box_info)

    return qty_box_info
