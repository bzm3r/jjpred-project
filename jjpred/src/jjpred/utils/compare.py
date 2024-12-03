"""Helpers for comparing actual dispatches with calculated dispatches. Mostly
not used now, as we tend to checks/checkdispatch.py."""

import sys
import polars as pl
from typing import Literal, NamedTuple
from jjpred.channel import Channel
from jjpred.utils.polars import binary_partition


def df_rows(df: pl.DataFrame) -> int:
    return df.shape[0]


class ExplainResult(NamedTuple):
    explained: pl.DataFrame
    unexplained: pl.DataFrame


def explain_diff(
    unexplained: pl.DataFrame, explanation: pl.Expr
) -> ExplainResult:
    parts = binary_partition(unexplained, explanation)
    return ExplainResult(parts[True], parts[False])


def sort_by_abs_max(df: pl.DataFrame, column: str, default: int) -> int:
    result = df[column].abs().max()
    if result is None:
        return default
    else:
        return int(result)  # type: ignore


def select_and_sort(
    df: pl.DataFrame,
    select_columns: list[str] | None = None,
    sort_by: list[str | pl.Expr] | None = None,
    descending: bool = False,
) -> pl.DataFrame:
    if select_columns is not None and len(select_columns) > 0:
        df = df.select(select_columns)

    if sort_by is not None and len(sort_by) > 0:
        df = df.sort(["sku", "category"] + Channel.members()).sort(
            sort_by, descending=descending
        )

    return df


def print_with_key(
    key: str,
    df: pl.DataFrame,
    select_columns: list[str] | None = None,
    sort_by: list[str | pl.Expr] | None = None,
    descending: bool = False,
):
    df = select_and_sort(
        df,
        select_columns=select_columns,
        sort_by=sort_by,
        descending=descending,
    )
    if len(df) > 0:
        print(f"{key}: {list(df["category"].unique().sort())}")
        sys.displayhook(df)


class Explanation(NamedTuple):
    name: str
    full_df: pl.DataFrame
    explanations: dict[str, pl.DataFrame]
    unexplained: pl.DataFrame

    def display_condensed(self):
        total_rows = df_rows(self.full_df)
        rows_per_explanation = {
            k: df_rows(v) for k, v in self.explanations.items()
        }
        explanation_rows = sum(rows_per_explanation.values())
        unexplained_rows = df_rows(self.unexplained)

        section_string = f"=== EXPLANATION {self.name} ==="
        print(section_string)
        print(f"{total_rows=}")

        for k, rows in rows_per_explanation.items():
            print(f"{k}={rows}")
        print("-" * len(section_string))
        print(f"{explanation_rows=}")
        print(f"{unexplained_rows=}")
        print(f"{(total_rows - unexplained_rows)=}")
        print(f"{(explanation_rows == (total_rows - unexplained_rows))=}")
        print(f"{(total_rows - explanation_rows)=}")
        print(f"{(total_rows == explanation_rows)=}")
        print("-" * len(section_string))
        print("=" * len(section_string))

    def display_full(
        self,
        select_columns: list[str] | None = None,
        sort_by: list[str | pl.Expr] | None = None,
        descending: bool = False,
    ):
        pl.Config().restore_defaults()
        pl.Config().set_tbl_rows(50)

        for key in self.explanations.keys():
            print_with_key(
                key,
                self.explanations[key],
                select_columns=select_columns,
                sort_by=sort_by,
                descending=descending,
            )

        print_with_key(
            "unexplained",
            self.unexplained,
            select_columns=select_columns,
            sort_by=sort_by,
            descending=descending,
        )

        pl.Config().restore_defaults()

    def display_by_category(self, key: str | Literal["unexplained"]):
        if key == "unexplained":
            focus_df = self.unexplained
        else:
            focus_df = self.explanations[key]

        dfs_per_category = {
            k[0]: v
            for k, v in focus_df.sort(
                "dispatch_delta", "delta_fraction"
            ).group_by("category")
        }

        for k, v in sorted(
            dfs_per_category.items(),
            key=lambda x: sort_by_abs_max(x[1], "dispatch_delta", -1),
            reverse=True,
        ):
            print(f"{k}:")
            sys.displayhook(
                v.sort(pl.col("dispatch_delta").abs(), descending=True)
            )


class NamedDf(pl.DataFrame):
    name: str

    def __init__(self, **name_and_df: pl.DataFrame):
        assert len(name_and_df) == 1
        for name, df in name_and_df.items():
            super().__init__(df)
            self.name = name


def create_explanations(
    name: str, full_df: pl.DataFrame, **kwargs: pl.Expr
) -> Explanation:
    explanations = {}
    unexplained = full_df
    for kw, expr in kwargs.items():
        explain_result = explain_diff(unexplained, expr)
        explanations[kw] = explain_result.explained
        unexplained = explain_result.unexplained
    return Explanation(name, full_df, explanations, unexplained)
