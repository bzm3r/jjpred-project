from typing import Any, Literal
import polars as pl
import altair as alt


def make_special_first_option(options: list[str]) -> list[str]:
    is_special = [x.startswith("_") for x in options]
    if any(is_special):
        return [
            x
            for x, x_is_special in zip(options, is_special, strict=True)
            if x_is_special
        ] + [
            x
            for x, x_is_special in zip(options, is_special, strict=True)
            if not x_is_special
        ]
    else:
        return options


def create_dropdown_selection(
    df: pl.DataFrame, column: str, default_choice: str | None = None
) -> alt.Parameter:
    options = df[column].unique().sort()
    return create_dropdown_selection_from_options(
        column,
        make_special_first_option(list(options)),
        default_choice=default_choice,
    )


def create_dropdown_selection_from_options(
    column: str, options: list[str], default_choice: str | None = None
) -> alt.Parameter:
    assert len(options) > 0
    name = f"{column.capitalize()}"
    selection = alt.selection_point(
        name + "_selector",
        fields=[column],
        bind=alt.binding_select(options=options, name=name),
        value=default_choice if default_choice is not None else options[0],
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


def create_legend_selection(
    df: pl.DataFrame,
    column: str,
    column_type: Literal["nominal", "ordinal", "quantitative", "temporal"],
    alt_color: Literal["lightgray", "transparent"],
) -> tuple[alt.Parameter, dict[str, Any]]:  # tuple[alt.Parameter, alt.Chart]:
    options = df[column].unique().sort()
    assert len(options) > 0
    name = f"{column.capitalize()}"

    selection = alt.selection_point(
        name + "_selector", fields=[column], bind="legend"
    )

    unselect_color = alt.condition(
        selection,
        alt.Color(column, type=column_type),
        alt.value(alt_color),
    )

    # legend = (
    #     alt.Chart(df)
    #     .mark_point()
    #     .encode(alt.Y(column).axis(orient="right"), color=unselect_color)
    # ).add_params(selection)

    return selection, unselect_color  # , legend
