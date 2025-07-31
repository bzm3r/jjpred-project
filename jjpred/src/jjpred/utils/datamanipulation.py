import polars as pl

from jjpred.channel import Channel
from jjpred.readsupport.utils import cast_standard, parse_channels
from jjpred.utils.polars import binary_partition_strict


def merge_wholesale(
    df: pl.DataFrame,
    group_by: list[str | pl.Expr],
    aggregations: list[pl.Expr],
) -> pl.DataFrame:
    wholesale_data, other_data = binary_partition_strict(
        df,
        pl.col.channel.cast(pl.String()).is_in(
            [
                Channel.parse(x).pretty_string_repr()
                for x in [
                    "Faire.com",
                    "Wholesale",
                    "Wholesale-CA",
                    "Wholesale-US",
                ]
            ],
        ),
    )

    assert "WHOLESALE" not in list(other_data["mode"].unique())

    df = other_data.select(df.columns).vstack(
        cast_standard(
            [df],
            parse_channels(
                wholesale_data.group_by(*group_by)
                .agg(*aggregations)
                .with_columns(
                    channel=pl.lit("Wholesale"),
                )
            )
            .drop("raw_channel")
            .select(df.columns),
        )
    )

    return df
