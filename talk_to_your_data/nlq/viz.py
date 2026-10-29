from typing import Literal

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
from pydantic import BaseModel, Field


class ChartSpec(BaseModel):
    chart: Literal["bar", "line", "scatter", "pie", "table"]
    x: str | None = None
    y: str | None = None
    color: str | None = None
    title: str = Field(default="", max_length=120)


class Answer(BaseModel):
    summary: str = Field(max_length=800)
    spec: ChartSpec


def is_valid(spec: ChartSpec, df: pd.DataFrame) -> bool:
    used = {c for c in (spec.x, spec.y, spec.color) if c}
    return (
        spec.chart != "table"
        and not df.empty
        and spec.x
        and spec.y
        and used <= set(df.columns)
        and pd.api.types.is_numeric_dtype(df[spec.y])
    )


def make_figure(spec: ChartSpec, df: pd.DataFrame) -> Figure | None:
    if not is_valid(spec, df):
        return None
    if spec.chart == "pie":
        return px.pie(df, names=spec.x, values=spec.y, title=spec.title)
    plot = {"bar": px.bar, "line": px.line, "scatter": px.scatter}[spec.chart]
    return plot(df, x=spec.x, y=spec.y, color=spec.color, title=spec.title)
