from typing import Literal
import polars as pl
import altair as alt


def create_dropdown_selection(df: pl.DataFrame, column: str) -> alt.Parameter:
    options = df[column].unique().sort()
    assert len(options) > 0
    name = f"{column.capitalize()}"
    selection = alt.selection_point(
        name + "_selector",
        fields=[column],
        bind=alt.binding_select(options=list(options), name=name),
        value=options[0],
    )

    return selection


def create_checkbox_selection(
    df: pl.DataFrame,
    column: str,
    alt_color: Literal["lightgray", "transparent"],
) -> tuple[alt.Parameter, alt.Chart]:
    options = df[column].unique().sort()
    assert len(options) > 0
    name = f"{column.capitalize()}"

    selection = alt.selection_point(
        name + "_selector",
        fields=[column],
    )

    unselect_color = alt.condition(
        selection, alt.Color(column).legend(None), alt.value(alt_color)
    )

    legend = (
        alt.Chart(df)
        .mark_point()
        .encode(alt.Y(column).axis(orient="right"), color=unselect_color)
    ).add_params(selection)

    return selection, legend
