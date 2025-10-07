"""Functions to read from the marketing configuration Excel file."""

from __future__ import annotations

from collections.abc import Sequence
import copy
from dataclasses import dataclass, field
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
from jjpred.analysisconfig import GeneralRefillConfigInfo, RefillConfigInfo
from jjpred.sku import (
    SKU_CATEGORY_PRINT_SIZE_REMAINDER,
    Sku,
)
from jjpred.structlike import FieldMeta, MemberType, StructLike
from jjpred.utils.datetime import Date, DateLike
import polars as pl
import polars.selectors as cs

from jjpred.utils.fileio import read_meta_info
from jjpred.utils.polars import (
    binary_partition_strict,
    concat_enum_extend_vstack_strict,
    find_dupes,
    sanitize_excel_extraction,
)
from jjpred.utils.typ import (
    ScalarOrList,
    create_assert_result,
    do_nothing,
)


def gen_config_path(analysis_input_folder: Path, date: DateLike) -> Path:
    """Generate the marketing configuration file path (e.g. ``category focus-20250929.xlsx``)
    of the required date."""
    date = Date.from_datelike(date)
    return analysis_input_folder.joinpath(
        f"category focus-{date.fmt_flat()}.xlsx"
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


@dataclass
class ConfigData:
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
    in_config_file: pl.DataFrame
    "Contains information about whether categories were found in the configuration file."

    def filter_channels(self, channel_df: pl.DataFrame) -> Self:
        channel_df = channel_df.select(Channel.members())
        self.refill = self.refill.join(channel_df, on=Channel.members())
        self.no_refill = self.no_refill.join(channel_df, on=Channel.members())
        self.min_keep = self.min_keep.join(channel_df, Channel.members())

        return self

    def extra_refill_info(
        self,
        active_sku_info: pl.DataFrame,
        extra_refill_info: Sequence[
            RefillConfigInfo | GeneralRefillConfigInfo
        ],
    ) -> Self:
        active_sku_info = active_sku_info.join(
            self.refill.select("sku", "a_sku", "refill_request")
            .group_by("sku", "a_sku")
            .agg(pl.col.refill_request.min().alias("min_refill_request")),
            on=["sku", "a_sku"],
            how="left",
        ).with_columns(pl.col.min_refill_request.fill_null(0))
        if len(extra_refill_info) > 0:
            normalized_extra_refill_info: list[RefillConfigInfo] = []
            for x in extra_refill_info:
                if isinstance(x, GeneralRefillConfigInfo):
                    normalized_extra_refill_info.append(
                        x.into_refill_config_info(active_sku_info)
                    )
                else:
                    normalized_extra_refill_info.append(x)

            result = copy.deepcopy(self)
            extra_refill_df = cast_standard(
                [active_sku_info],
                parse_channels(
                    pl.from_dicts(
                        x.as_dict() for x in normalized_extra_refill_info
                    )
                    .explode("channel")
                    .explode("sku")
                ).drop("raw_channel", "channel"),
            ).join(
                active_sku_info.select(
                    "sku", "a_sku", *SKU_CATEGORY_PRINT_SIZE_REMAINDER
                ),
                on=["sku"],
                how="left",
            )
            assert len(extra_refill_df.filter(pl.col.category.is_null())) == 0

            new_refill_df = (
                concat_enum_extend_vstack_strict(
                    [
                        result.refill.filter(
                            ~pl.struct("sku", *Channel.members()).is_in(
                                extra_refill_df.select(
                                    pl.struct("sku", *Channel.members()).alias(
                                        "id"
                                    )
                                )["id"]
                            )
                        ),
                        extra_refill_df,
                    ]
                )
                .group_by(*ALL_SKU_AND_CHANNEL_IDS)
                .agg(pl.col.refill_request.max())
            )

            find_dupes(
                new_refill_df, ["sku"] + Channel.members(), raise_error=True
            )

            result.refill = new_refill_df

            return result

        return self

    @classmethod
    def create(
        cls,
        refill: pl.DataFrame,
        min_keep: pl.DataFrame,
        category_season: pl.DataFrame,
        in_config_file: pl.DataFrame,
    ) -> Self:
        """Generate from ``refill`` and ``min_keep`` dataframes."""
        refill, no_refill = binary_partition_strict(
            refill, pl.col("refill_request").ge(0)
        )
        return cls(
            refill, no_refill, min_keep, category_season, in_config_file
        )


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
                gen_config_path(
                    analysis_input_folder,
                    date,
                ),
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


def filter_channel_info(
    config_df: pl.DataFrame, relevant_channels: pl.DataFrame
) -> pl.DataFrame:
    """Filter channel information in a dataframe."""
    return config_df.drop(
        [x for x in ["channel", "raw_channel"] if x in config_df.columns]
    ).join(relevant_channels, on=Channel.members())
    # if "channel" in config_df.columns:
    #     return config_df.join(
    #         relevant_channels, on="channel", how="left"
    #     ).drop("channel")
    # else:
    #     if "channel" in relevant_channels:
    #         relevant_channels = relevant_channels.drop("channel")
    #     return config_df.join(
    #         relevant_channels,
    #         how="cross",
    #         on=None,
    #     )


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
    analysis_defn: AnalysisDefn, channels: Sequence[str | Channel]
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


def read_config(analysis_defn: AnalysisDefn) -> ConfigData:
    """Read configuration information from the marketing configuration Excel
    file.

    It assumes that only Amazon.com/Amazon.ca data is present.
    """
    channel_info = parse_channels(
        pl.DataFrame(
            pl.Series(
                "channel",
                [
                    "Amazon.com",
                    "Amazon.ca",
                ],
            )
        )
    ).drop("raw_channel", "channel")
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
    sku_part_info = sku_part_info.join(channel_info, how="cross")
    # sys.displayhook(sku_part_info)

    use_columns = [
        RelevantColumn(0, "category"),
        RelevantColumn(3, "Amazon.com"),
        RelevantColumn(7, "Amazon.ca"),
    ]
    config_date = analysis_defn.config_date

    if config_date is None:
        raise ValueError(f"{analysis_defn.config_date=}")

    refill_request = extract_df(
        ANALYSIS_INPUT_FOLDER,
        config_date,
        use_columns,
    )

    in_config_file = (
        refill_request.select("category")
        .with_columns(in_config_file=pl.lit(True))
        .group_by("category")
        .agg(
            pl.col.in_config_file.all(),
            pl.col.in_config_file.sum().alias("num_config_file_appearances"),
        )
    )

    refill_request = (
        refill_request.unpivot(
            index="category", on=["Amazon.com", "Amazon.ca"]
        )
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

    assert (
        len(in_config_file.filter(pl.col.num_config_file_appearances.gt(1)))
        == 0
    )

    # sys.displayhook(refill)
    refill_request = filter_channel_info(
        parse_channels(refill_request).drop("raw_channel", "channel"),
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
        extract_df(
            ANALYSIS_INPUT_FOLDER,
            config_date,
            use_columns,
        )
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

    assert len(min_keep) == (len(refill_request) / len(channel_info)), (
        create_assert_result(
            min_keep=len(min_keep),
            refill_request=len(refill_request),
            channel_info=len(channel_info),
        )
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

    min_keep = (
        min_keep.join(
            active_sku_info.select(ALL_SKU_IDS),
            on=relevant_ids,
            # how = default (inner), because we want to filter out those category + size
            # + print that do not have a matching active_sku (i.e. they are inactive)
            # RHS: some items with the same category/print/size have multiple SKUs
            validate="1:m",
        )
        .select(ALL_SKU_IDS + ["min_keep"])
        .join(channel_info, how="cross")
    )
    # sys.displayhook(min_keep)

    assert len(min_keep) == len(refill_request)

    use_columns = [
        RelevantColumn(0, "category"),
        RelevantColumn(1, "season"),
    ]
    category_season = cast_standard(
        [active_sku_info],
        extract_df(
            ANALYSIS_INPUT_FOLDER,
            config_date,
            use_columns,
        ),
    )

    in_config_file = (
        active_sku_info.select("category")
        .unique()
        .join(
            cast_standard(
                [active_sku_info],
                in_config_file.select("category", "in_config_file"),
            ),
            on=["category"],
            how="left",
        )
        .with_columns(pl.col.in_config_file.fill_null(False))
    )

    return ConfigData.create(
        refill_request, min_keep, category_season, in_config_file
    )
