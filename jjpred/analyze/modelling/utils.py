from collections import defaultdict
from dataclasses import dataclass
import polars as pl
from jjpred.analysisdefn import AnalysisDefn
from jjpred.countryflags import CountryFlags
from jjpred.database import DataBase
from jjpred.datagroups import ALL_SKU_IDS, WHOLE_SKU_IDS
from jjpred.sku import Sku
from jjpred.structlike import MemberType
from jjpred.utils.fileio import read_meta_info
from jjpred.utils.typ import as_list, ScalarOrList, as_polars_type


def get_analysis_defn_and_db(
    analysis_defn_or_db: AnalysisDefn | DataBase,
    read_db_from_disk: bool = True,
) -> tuple[AnalysisDefn, DataBase]:
    """Given an analysis definition OR database, get an analysis definition
    AND database."""
    if isinstance(analysis_defn_or_db, AnalysisDefn):
        analysis_defn = analysis_defn_or_db
        db = DataBase(analysis_defn, read_from_disk=read_db_from_disk)
    else:
        db = analysis_defn_or_db
        analysis_defn = db.analysis_defn

    return analysis_defn, db


def create_agg_label_default_dict(
    agg_labels: dict[str, str], default_value: str = "_ALL_"
) -> defaultdict[str, str]:
    result = defaultdict(lambda: default_value)
    for k, v in agg_labels.items():
        result[k] = v
    return result


def sum_quantity_in_order(
    analysis_defn: AnalysisDefn,
    df: pl.DataFrame,
    cols_to_sum: ScalarOrList[str],
    ordered_sum_by_cols: list[str],
    aggregated_label_for_sum_by_cols: defaultdict[str, str],
) -> pl.DataFrame:
    """Sum given columns in order over the grouping columns:
    ``category``, ``channel``, ``date`` and each one of ``ordered_sum_by_cols``
    in reverse sequence.

    For example: if ``ordered_sum_by_cols`` is ``["print", "size"]``, then first
    we aggregate grouping by ``["category", "channel", "date", "print"]`` and
    then ``["category", "channel", "date", "size"]``.

    Those of ``category``, ``channel`` and ``date`` also in given additional
    columns to aggregate by, are not used as index columns."""

    cols_to_sum = as_list(cols_to_sum)

    df = (
        df.select(
            ["channel"] + WHOLE_SKU_IDS + ["category", "date"] + cols_to_sum
        )
        .join(
            read_meta_info(analysis_defn, "all_sku").select(
                [c for c in ALL_SKU_IDS if c != "category"]
            ),
            on=WHOLE_SKU_IDS,
        )
        .select(
            ["channel"]
            + Sku.members(MemberType.SECONDARY)
            + ["date"]
            + cols_to_sum
        )
    )

    all_minimum_index_cols = ["category", "channel", "date"]
    minimum_index_cols = [
        x for x in all_minimum_index_cols if x not in ordered_sum_by_cols
    ]

    for agg_by_col in reversed(ordered_sum_by_cols):
        this_index_cols = minimum_index_cols + [
            ic for ic in ordered_sum_by_cols if ic != agg_by_col
        ]

        df = df.vstack(
            df.group_by(this_index_cols)
            .agg(pl.col(agg_col).sum() for agg_col in cols_to_sum)
            .with_columns(
                pl.lit(
                    aggregated_label_for_sum_by_cols[agg_by_col],
                    dtype=df[agg_by_col].dtype,
                ).alias(agg_by_col)
            )
            .select(df.columns)
        )

    missing_months_filter = pl.col.date.ge(analysis_defn.date.as_polars_date())
    df.with_columns(
        pl.when(missing_months_filter)
        .then(None)
        .otherwise(pl.col(agg_col))
        .alias(agg_col)
        for agg_col in cols_to_sum
    )

    return df


def enum_extend_vstack(
    df: pl.DataFrame, other_df: pl.DataFrame
) -> pl.DataFrame:
    assert sorted(df.columns) == sorted(other_df.columns)

    for column in df.columns:
        if isinstance(df[column].dtype, pl.Enum):
            categories = as_polars_type(df[column].dtype, pl.Enum).categories
            other_categories = as_polars_type(
                other_df[column].dtype, pl.Enum
            ).categories
            if (not (len(categories) == len(other_categories))) or (
                not categories.eq(other_categories).all()
            ):
                combined_categories = (
                    categories.extend(other_categories).unique().sort()
                )
                df = df.cast({column: pl.Enum(combined_categories)})
                other_df = other_df.cast(
                    {column: pl.Enum(combined_categories)}
                )

    return df.vstack(other_df.select(df.columns))


@dataclass
class ChannelFilter:
    description: str
    expression: pl.Expr


# @dataclass
# class ChannelFilters:
#     filters: list[ChannelFilter]

#     def all_descriptions(self) -> list[str]:
#         return [filter.description for filter in self.filters]


# KNOWN_CHANNEL_FILTERS = Literal["_NA_MAJOR_RETAIL_", "_ALL_WHOLESALE_"]
# CHANNEL_FILTER_LIBRARY: dict[KNOWN_CHANNEL_FILTERS, pl.Expr] = {
#     "_NA_MAJOR_RETAIL_": (pl.col.mode == "RETAIL")
#     & (pl.col.country_flag.and_(int(CountryFlags.CA | CountryFlags.US)) > 0)
#     & (pl.col.platform.is_in(["Amazon", "JanAndJul"])),
#     "_ALL_WHOLESALE_": pl.col.mode.eq("WHOLESALE"),
# }

NA_MAJOR_RETAIL = ChannelFilter(
    "_NA_MAJOR_RETAIL_",
    (pl.col.mode == "RETAIL")
    & (pl.col.country_flag.and_(int(CountryFlags.CA | CountryFlags.US)) > 0)
    & (pl.col.platform.is_in(["Amazon", "JanAndJul"])),
)

WHOLESALE_FILTER = ChannelFilter(
    "_ALL_WHOLESALE_", pl.col.mode.eq("WHOLESALE")
)

EU_RETAIL = ChannelFilter(
    "_EU_RETAIL_",
    (pl.col.mode == "RETAIL")
    & (
        pl.col.country_flag.and_(
            int(CountryFlags.DE | CountryFlags.EU | CountryFlags.UK)
        )
        > 0
    ),
)

KNOWN_CHANNEL_FILTERS = [NA_MAJOR_RETAIL, WHOLESALE_FILTER, EU_RETAIL]
