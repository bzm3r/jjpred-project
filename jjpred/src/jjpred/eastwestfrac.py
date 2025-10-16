"""Logic for calculating JJWeb East/West sales fractions, for 3PL dispatch."""

import polars as pl
import polars.selectors as cs
from jjpred.analysisdefn import RefillDefn
from jjpred.channel import Channel, SubCountry
from jjpred.structlike import MemberType
from jjpred.utils.datetime import first_day


def calculate_fractions(
    history_df: pl.DataFrame, index_cols: list[str]
) -> pl.DataFrame:
    frac_df = (
        history_df.group_by(*index_cols, "sub_country")
        .agg(pl.col.sales.sum())
        .pivot("sub_country", index=index_cols, values=["sales"])
        .with_columns(total=pl.col.EAST + pl.col.WEST)
        .with_columns(
            [
                pl.when(pl.col.total.gt(0))
                .then(pl.col(x.name) / pl.col.total)
                .otherwise(0.5)
                .alias(x.name.lower() + "_frac")
                for x in SubCountry
                if x.name != SubCountry.ALL.name
            ]
        )
    )

    return frac_df


def calculate_east_west_fracs(
    analysis_defn: RefillDefn, history_df: pl.DataFrame
) -> pl.DataFrame:
    relevant_history = (
        history_df.filter(
            pl.struct(Channel.members()).eq(
                Channel.parse("jjweb ca east").as_dict()
            )
            | pl.struct(Channel.members()).eq(
                Channel.parse("jjweb ca west").as_dict()
            )
        )
        .filter()
        .drop(
            *[
                x
                for x in Channel.members(MemberType.META)
                if x != "sub_country"
            ]
        )
        .filter(
            pl.col.date.ge(
                first_day(analysis_defn.dispatch_date)
                .as_polars_date()
                .dt.offset_by("-1y")
            ),
        )
    )

    if len(relevant_history) == 0:
        return pl.DataFrame()

    sku_specific = calculate_fractions(relevant_history, ["a_sku", "category"])

    category_specific = calculate_fractions(relevant_history, ["category"])

    east_west_frac_df = (
        sku_specific.select(
            "a_sku",
            "category",
            cs.by_name(
                ["EAST", "WEST", "east_frac", "west_frac", "total"]
            ).name.suffix("_sku"),
        )
        .join(
            category_specific.select(
                "category",
                cs.by_name(
                    ["EAST", "WEST", "total", "east_frac", "west_frac"]
                ).name.suffix("_category"),
            ),
            on=["category"],
            validate="m:1",
        )
        .with_columns(
            [
                pl.when(
                    pl.col.total_category.gt(0)
                    & (pl.col.total_sku / pl.col.total_category).lt(0.5)
                    & pl.col.total_sku.lt(3)
                )
                .then(pl.col(x.name.lower() + "_frac_category"))
                .otherwise(pl.col(x.name.lower() + "_frac_sku"))
                .alias(x.name.lower() + "_frac")
                for x in SubCountry
                if x != SubCountry.ALL
            ]
        )
    )

    return east_west_frac_df
