from math import inf
import polars as pl


def bin_months(series: pl.Series) -> pl.DataFrame:
    return (
        series.hist(bins=[x + 1 + 0.1 for x in range(12)])
        .filter(pl.col.breakpoint.lt(inf))
        .with_columns(
            pl.col.breakpoint.floor().cast(pl.Int8()).alias(series.name)
        )
        .select(series.name, "count")
    )


def bin_months_over(
    df: pl.DataFrame, month_col: str, over_col: str
) -> pl.DataFrame:
    # initialize an empty dataframe
    category_binned_df = pl.DataFrame()

    for x in df[over_col].unique():
        # repeat the binning logic from earlier, except on a dataframe filtered for
        # the particular category we are iterating over
        binned_df = (
            df.filter(pl.col(over_col).eq(x))  # <--- the filter
            .select(
                pl.col(month_col)
                .hist(
                    bins=[x + 1 for x in range(11)],
                    include_breakpoint=True,
                )
                .alias("binned"),
            )
            .unnest("binned")
            .with_columns(
                pl.col.breakpoint.map_elements(
                    lambda x: 12 if x == inf else x, return_dtype=pl.Float64()
                )
                .cast(pl.Int8())
                .alias(month_col)
            )
            .drop("breakpoint")
            .select(month_col, "count")
            .with_columns(pl.lit(x).cast(df[over_col].dtype).alias(over_col))
        )
        # finally, vstack ("append") the resulting dataframe
        category_binned_df = category_binned_df.vstack(binned_df)
    return category_binned_df
