"""Functions to read from the marketing configuration Excel file."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import NamedTuple, Self
from jjpred.analysisdefn import AnalysisDefn
from jjpred.channel import Channel
from jjpred.datagroups import ALL_SKU_AND_CHANNEL_IDS, ALL_SKU_IDS
from jjpred.globalpaths import ANALYSIS_INPUT_FOLDER
from jjpred.parse.patternmatch import (
    PatternMatchResult,
    PatternMatcher,
)
from jjpred.readsupport.utils import cast_standard, parse_channels
from jjpred.sku import Sku
from jjpred.structlike import FieldMeta, MemberType, StructLike
from jjpred.utils.datetime import Date, DateLike
import polars as pl
import polars.selectors as cs

from jjpred.utils.fileio import read_meta_info
from jjpred.utils.polars import (
    binary_partition_strict,
    sanitize_excel_extraction,
)
from jjpred.utils.typ import ScalarOrList, create_assert_result, do_nothing


def gen_config_path(analysis_input_folder: Path, date: DateLike) -> Path:
    """Generate the marketing configuration file path (``2023 category focus``)
    of the required date."""
    return analysis_input_folder.joinpath(
        f"2023 category focus-{Date.from_datelike(date).fmt_flat()}.xlsx"
    )


class RelevantColumn(NamedTuple):
    """Integer index of relevant columns from the config file, and what they
    should be renamed to."""

    ix: int
    """Integer index of relevant column."""
    rename_to: str
    """Name the column should be renamed to in a dataframe."""


class ConfigStrMatcher(PatternMatcher):
    """Used to match strings of the type ``SIZE-1T|2T``, ``PRINT-WPF|BST`` or
    ``QTY-5``."""

    name: str = "config_str_matcher"

    def apply(
        self, target_strings: ScalarOrList[str]
    ) -> PatternMatchResult | None:
        if isinstance(target_strings, list) and len(target_strings) > 0:
            raise ValueError(f"Cannot parse {target_strings} as ConfigStr")
        elif isinstance(target_strings, str):
            string = target_strings
        else:
            string = target_strings[0]

        cs_parts = [y.strip() for x in string.split(",") for y in x.split(";")]
        parse_results = PatternMatchResult({})
        for part in cs_parts:
            if part.startswith("SIZE-"):
                parse_results["size"] = [
                    x
                    for x in part.removeprefix("SIZE-").strip().split("|")
                    if len(x) > 0
                ]
            elif part.startswith("PRINT-"):
                parse_results["print"] = [
                    x
                    for x in part.removeprefix("PRINT-").strip().split("|")
                    if len(x) > 0
                ]
            elif part.startswith("QTY-"):
                parse_results["qty"] = int(part.removeprefix("QTY-").strip())
            elif part.startswith("QTY -"):
                parse_results["qty"] = int(part.removeprefix("QTY -").strip())
            elif part == "NO REFILL":
                parse_results["qty"] = -1
            else:
                raise ValueError(f"No logic to parse: {part}")

        return parse_results

    @classmethod
    def combine(cls, left: Self, right: Self) -> Self:
        raise ValueError(f"Cannot add: {left=} + {right=}")


class MinKeepQty(
    StructLike,
    matcher=ConfigStrMatcher(),
    joiner=" ",
):
    """Data for the minimum quantity to reserve in the warehouse for the
    relevant size and prints, as read from the config file."""

    size: list[str] = field(
        default_factory=list,
        metadata=FieldMeta(
            MemberType.PRIMARY, do_nothing, pl.List(pl.String())
        ),
    )
    """Relevant sizes."""
    print: list[str] = field(
        default_factory=list,
        metadata=FieldMeta(
            MemberType.PRIMARY, do_nothing, pl.List(pl.String())
        ),
    )
    """Relevant prints."""
    qty: int | None = field(
        default=None,
        metadata=FieldMeta(
            MemberType.PRIMARY,
            lambda x: int(x) if x is not None else None,
            pl.Int64(),
        ),
    )
    """Minimum quantity to keep (defaults to ``None``.)"""

    @classmethod
    def from_dict(cls, x: dict[str, str]) -> Self:
        return cls(
            **(
                cls.field_defaults
                | {
                    k: x[k]
                    for k in cls.fields_by_cutoff[MemberType.SECONDARY]
                    if x.get(k) is not None
                }
            )
        )

    def __hash__(self) -> int:
        return super().__hash__()


class RefillParams(
    StructLike,
    matcher=ConfigStrMatcher(),
    joiner=" ",
):
    """Data for the minimum dispatch ('refill quantity') to send for the
    relevant size and prints, as read from the config file."""

    size: list[str] = field(
        default_factory=list,
        compare=False,
        metadata=FieldMeta(
            MemberType.PRIMARY, do_nothing, pl.List(pl.String())
        ),
    )
    """Relevant sizes."""

    print: list[str] = field(
        default_factory=list,
        compare=False,
        metadata=FieldMeta(
            MemberType.PRIMARY, do_nothing, pl.List(pl.String())
        ),
    )
    """Relevant prints."""

    qty: int = field(
        default=0,
        compare=False,
        metadata=FieldMeta(MemberType.PRIMARY, int, pl.Int64()),
    )
    """Minimum dispatch to dispatch for these sizes and prints.

    Defaults to ``0``."""

    @classmethod
    def from_dict(cls, x: dict[str, str]) -> Self:
        return cls(
            **(
                cls.field_defaults
                | {
                    k: x[k]
                    for k in cls.fields_by_cutoff[MemberType.SECONDARY]
                    if x.get(k) is not None
                }
            )
        )

    def __hash__(self) -> int:
        return super().__hash__()


class ConfigData(NamedTuple):
    """Configuration data read from the marketing configuration file."""

    refill: pl.DataFrame
    """SKUs with some refill quantity set (possibly ``0``)."""
    no_refill: pl.DataFrame
    """SKUs marked ``NO_REFILL``."""
    min_keep: pl.DataFrame
    """SKUs with some minimum keep-in-warehouse quantity set (those that do not
    have any values set have data ``None``.)"""
    category_season: pl.DataFrame
    """Configured category season.

    This can affect the prediction type (E vs. PO based) for items."""

    @classmethod
    def create(
        cls,
        refill: pl.DataFrame,
        min_keep: pl.DataFrame,
        category_season: pl.DataFrame,
    ) -> Self:
        """Generate from ``refill`` and ``min_keep`` dataframes."""
        refill, no_refill = binary_partition_strict(
            refill, pl.col("refill_request").ge(0)
        )
        return cls(refill, no_refill, min_keep, category_season)


def extract_df(
    analysis_input_folder: Path,
    date: DateLike,
    use_columns: list[RelevantColumn],
) -> pl.DataFrame:
    """Extract a dataframe from an excel file, given relevant columns."""
    rename_map = dict((f"column_{x.ix + 1}", x.rename_to) for x in use_columns)
    return (
        sanitize_excel_extraction(
            pl.read_excel(
                gen_config_path(analysis_input_folder, date),
                sheet_name="CONFIG",
                read_options={
                    "skip_rows": 2,
                    "header_row": None,
                },
            )
        )
        .select(rename_map.keys())
        .rename(rename_map)
    )


def df_sequence_explode(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Explode list/array columns in a dataframe in sequence, if the length of
    elements in each column are not necessarily the same."""
    for column in columns:
        if column in df.columns:
            if isinstance(df[column].dtype, pl.List) or isinstance(
                df[column].dtype, pl.Array
            ):
                df = df.explode(column)
    return df


def attach_channel_info(
    config_df: pl.DataFrame, relevant_channels: pl.DataFrame
) -> pl.DataFrame:
    """Attach channel information to a dataframe."""
    if "channel" in config_df.columns:
        return config_df.join(
            relevant_channels, on="channel", how="left"
        ).drop("channel")
    else:
        if "channel" in relevant_channels:
            relevant_channels = relevant_channels.drop("channel")
        return config_df.join(
            relevant_channels,
            how="cross",
            on=None,
        )


def attach_default_info(
    df: pl.DataFrame,
    default_df: pl.DataFrame,
    relevant_sku_parts: list[str],
    column: str,
):
    """Attach default value information to a dataframe."""
    column_default = f"{column}_default"
    return (
        df_sequence_explode(
            default_df,
            relevant_sku_parts,
        )
        .join(
            df_sequence_explode(
                df,
                relevant_sku_parts,
            ),
            on=cs.expand_selector(df, cs.exclude(column)),
            how="left",
        )
        .with_columns(pl.coalesce(column, column_default))
        .drop(column_default)
    )


def create_config_default_column(
    sku_part_info: pl.DataFrame,
    column: str,
    default_value: int | None,
) -> pl.DataFrame:
    """Create a default-value column (named ``{column}_default``) set to the
    default value provided."""
    return sku_part_info.with_columns(
        pl.lit(default_value, dtype=pl.Int64()).alias(f"{column}_default")
    )


def aggregate_relevant_sku_parts(
    df: pl.DataFrame, relevant_sku_parts: list[str]
) -> pl.DataFrame:
    """Aggregate relevant SKU parts into a list."""
    return (
        df.group_by(cs.exclude(relevant_sku_parts))
        .agg(pl.col(x) for x in relevant_sku_parts)
        .with_columns(pl.col(x).list.drop_nulls() for x in relevant_sku_parts)
    )


def cast_and_reaggregate_sku_parts(
    active_sku_info, df: pl.DataFrame, relevant_sku_parts: list[str]
) -> pl.DataFrame:
    """Temporarily explode relevant sku parts, re-cast them into standard data
    types based on the active sku information dataframe, and then re-aggregate
    them."""
    return aggregate_relevant_sku_parts(
        cast_standard(
            [active_sku_info],
            df_sequence_explode(
                df,
                relevant_sku_parts,
            ),
        ),
        relevant_sku_parts,
    )


def fill_out_sku_parts(
    df: pl.DataFrame,
    sku_part_info: pl.DataFrame,
    relevant_sku_parts: list[str],
    index_cols: list[str],
):
    """Replace empty lists of SKU parts with a list containing all possible
    values for that SKU."""
    return (
        df.join(
            sku_part_info,
            on=index_cols,
            suffix="_default",
        )
        .with_columns(
            pl.when(pl.col(x).list.len().gt(0)).then(pl.col(x)).otherwise(None)
            for x in relevant_sku_parts
        )
        .with_columns(
            pl.coalesce(x, f"{x}_default") for x in relevant_sku_parts
        )
        .drop(f"{x}_default" for x in relevant_sku_parts)
    )


def generate_channel_df(
    analysis_defn: AnalysisDefn, channels: list[str | Channel]
) -> pl.DataFrame:
    """Generate the channel information dataframe."""
    channel_info = read_meta_info(analysis_defn, "channel")
    relevant_channels = pl.Series(
        "raw_channel",
        [
            ch if isinstance(ch, str) else ch.str_repr(MemberType.PRIMARY)
            for ch in channels
        ],
    )
    return cast_standard(
        [channel_info],
        pl.DataFrame(relevant_channels)
        .with_columns(
            channel_info=pl.col.channel.map_elements(
                Channel.map_polars,
                return_dtype=Channel.intermediate_polars_type_struct(),
            )
        )
        .unnest("channel_info"),
    )


def read_config(
    analysis_defn: AnalysisDefn,
    relevant_channels: list[str | Channel] = ["Amazon.com", "Amazon.ca"],
) -> ConfigData:
    """Read configuration information from the marketing configuration Excel
    file.

    It assumes that only Amazon.com/Amazon.ca data is present, but this can be
    configured.
    """
    channel_info = parse_channels(
        pl.DataFrame(pl.Series("channel", relevant_channels))
    ).drop("raw_channel")
    # channel_info = generate_channel_df(analysis_defn, relevant_channels)
    # sys.displayhook(channel_info)

    active_sku_info = read_meta_info(analysis_defn, "active_sku")

    relevant_sku_parts = [
        "print",
        "size",
    ]
    sku_part_info = (
        active_sku_info.select(ALL_SKU_IDS)
        .unique()
        .group_by("category")
        .agg(pl.col(x) for x in relevant_sku_parts)
        .with_columns(pl.col(x).list.unique() for x in relevant_sku_parts)
    )
    sku_part_info = attach_channel_info(
        sku_part_info,
        channel_info,
    )
    # sys.displayhook(sku_part_info)

    use_columns = [
        RelevantColumn(0, "category"),
        RelevantColumn(3, "Amazon.com"),
        RelevantColumn(7, "Amazon.ca"),
    ]
    config_date = analysis_defn.config_date

    if config_date is None:
        raise ValueError(f"{analysis_defn.config_date=}")

    refill_request = (
        extract_df(ANALYSIS_INPUT_FOLDER, config_date, use_columns)
        .unpivot(index="category", on=["Amazon.com", "Amazon.ca"])
        .rename({"variable": "channel", "value": "refill_params"})
        .with_columns(
            pl.col("refill_params").map_elements(
                RefillParams.map_polars,
                skip_nulls=False,
                return_dtype=RefillParams.intermediate_polars_type_struct(),
            ),
        )
        .unnest("refill_params")
        .rename({"qty": "refill_request"})
    )

    # sys.displayhook(refill)
    refill_request = attach_channel_info(
        refill_request,
        channel_info,
    )
    # sys.displayhook(refill)

    refill_request = cast_and_reaggregate_sku_parts(
        active_sku_info, refill_request, relevant_sku_parts
    )
    # sys.displayhook(refill)

    refill_request = fill_out_sku_parts(
        refill_request,
        sku_part_info,
        relevant_sku_parts,
        ["category"] + Channel.members(),
    )
    # sys.displayhook(refill_request)

    default_refill = create_config_default_column(
        sku_part_info, "refill_request", -1
    )
    # sys.displayhook(default_refill)
    # sys.displayhook(
    #     default_refill.filter(
    #         pl.col.category.eq(focus_category), pl.col.channel.eq(focus_channel)
    #     )
    # )
    refill_request = attach_default_info(
        refill_request,
        default_refill,
        relevant_sku_parts,
        "refill_request",
    )
    # sys.displayhook(refill_request)
    # sys.displayhook(
    #     struct_filter(
    #         refill.filter(
    #             pl.col.category.eq(focus_category),
    #         ),
    #         Channel.parse(focus_channel),
    #     ).filter(pl.col("refill_request").ge(0))
    # )
    use_columns = [
        RelevantColumn(0, "category"),
        RelevantColumn(10, "min_keep"),
    ]
    min_keep = (
        extract_df(ANALYSIS_INPUT_FOLDER, config_date, use_columns)
        .filter(pl.col("min_keep").is_not_null())
        .with_columns(
            pl.col("min_keep").map_elements(
                MinKeepQty.map_polars,
                skip_nulls=False,
                return_dtype=MinKeepQty.intermediate_polars_type_struct(),
            )
        )
        .unnest("min_keep")
        .rename({"qty": "min_keep"})
    )

    min_keep = cast_and_reaggregate_sku_parts(
        active_sku_info, min_keep, relevant_sku_parts
    )

    sku_part_info = sku_part_info.drop(Channel.members()).unique()

    min_keep = fill_out_sku_parts(
        min_keep, sku_part_info, relevant_sku_parts, ["category"]
    )

    default_min_keep = create_config_default_column(
        sku_part_info, "min_keep", None
    )

    min_keep = attach_default_info(
        min_keep,
        default_min_keep,
        relevant_sku_parts,
        "min_keep",
    )
    # sys.displayhook(min_keep)

    assert len(min_keep) == (
        len(refill_request) / len(channel_info)
    ), create_assert_result(
        min_keep=len(min_keep),
        refill_request=len(refill_request),
        channel_info=len(channel_info),
    )

    relevant_ids = [
        x for x in Sku.members(MemberType.SECONDARY) if x != "sku_remainder"
    ]

    refill_request = refill_request.join(
        active_sku_info.select(ALL_SKU_IDS),
        on=relevant_ids,
        # how = default (inner), because we want to filter out those category + size
        # + print that do not have a matching active_sku (i.e. they are inactive)
        # LHS: each SKU is associated with two channels
        # RHS: some items with the same category/print/size have multiple SKUs
        validate="m:m",
    ).select(ALL_SKU_AND_CHANNEL_IDS + ["refill_request"])

    min_keep = min_keep.join(
        active_sku_info.select(ALL_SKU_IDS),
        on=relevant_ids,
        # how = default (inner), because we want to filter out those category + size
        # + print that do not have a matching active_sku (i.e. they are inactive)
        # RHS: some items with the same category/print/size have multiple SKUs
        validate="1:m",
    ).select(ALL_SKU_IDS + ["min_keep"])
    # sys.displayhook(min_keep)

    assert len(min_keep) == (len(refill_request) / len(channel_info))

    use_columns = [
        RelevantColumn(0, "category"),
        RelevantColumn(1, "season"),
    ]
    category_season = cast_standard(
        [active_sku_info],
        extract_df(ANALYSIS_INPUT_FOLDER, config_date, use_columns),
    )

    return ConfigData.create(refill_request, min_keep, category_season)
