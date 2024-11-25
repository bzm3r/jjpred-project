import altair as alt

type AltairChart = (
    alt.Chart
    | alt.VConcatChart
    | alt.HConcatChart
    | alt.ConcatChart
    | alt.LayerChart
)
