"""All Pydantic models: settings, agent.yaml schema, tool arguments and the agent output contract.

Named pydantic_models.py (not pydantic.py) so it can never shadow the pydantic package.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from .sql_guard import validate_sql

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_\-]+$")]
ChartKind = Literal["line", "bar", "scatter", "area", "histogram", "map"]
ToolName = Literal["describe_table", "run_sql", "submit_answer"]

DEFAULT_AGENT_YAML = Path(__file__).resolve().parent.parent / "agent.yaml"
_MEASUREMENT = re.compile(r"\d+[.,]\d+|\d+\s?(°|%|mm|cm|km|m/s|hpa)", re.IGNORECASE)


# --- Runtime settings (environment) ---------------------------------------------------
class AppSettings(BaseSettings):
    """Environment variables. The Databricks App injects these from databricks.yml."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    databricks_warehouse_id: str = Field(min_length=1)
    llm_endpoint: str | None = None
    weather_catalog: Identifier | None = None
    weather_schema: Identifier | None = None
    agent_config_path: Path = DEFAULT_AGENT_YAML


# --- agent.yaml -----------------------------------------------------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSettings(_Strict):
    endpoint: str = Field(min_length=1)
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(2000, ge=256, le=8192)


class DataSettings(_Strict):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    catalog: Identifier
    schema_name: Identifier = Field(alias="schema")
    units_hint: str = "metric units"


class Limits(_Strict):
    max_steps: int = Field(8, ge=2, le=20)
    max_rows: int = Field(5000, ge=10, le=100_000)
    preview_rows: int = Field(30, ge=1, le=200)
    query_timeout_s: int = Field(120, ge=10, le=600)
    max_output_repairs: int = Field(2, ge=0, le=5)
    history_messages: int = Field(6, ge=0, le=20)


class OutputRules(_Strict):
    answer_max_chars: int = Field(1200, ge=100, le=5000)
    max_follow_ups: int = Field(3, ge=0, le=5)
    chart_kinds: list[ChartKind] = Field(min_length=1)
    require_query_for_numbers: bool = True


class AgentConfig(_Strict):
    name: str
    version: str
    description: str
    model: ModelSettings
    data: DataSettings
    limits: Limits
    tools: list[ToolName] = Field(min_length=1)
    output: OutputRules
    instructions: str

    @field_validator("instructions")
    @classmethod
    def _has_schema_placeholder(cls, value: str) -> str:
        if "{schema_overview}" not in value:
            raise ValueError("instructions must contain the {schema_overview} placeholder")
        return value

    @model_validator(mode="after")
    def _final_answer_tool_enabled(self) -> AgentConfig:
        if "submit_answer" not in self.tools:
            raise ValueError("tools must include submit_answer (it carries the validated output)")
        return self

    @classmethod
    def load(cls, settings: AppSettings) -> AgentConfig:
        """Read agent.yaml, apply environment overrides, then validate everything at once."""
        raw: dict[str, Any] = yaml.safe_load(settings.agent_config_path.read_text(encoding="utf-8"))
        if settings.llm_endpoint:
            raw["model"]["endpoint"] = settings.llm_endpoint
        if settings.weather_catalog:
            raw["data"]["catalog"] = settings.weather_catalog
        if settings.weather_schema:
            raw["data"]["schema"] = settings.weather_schema
        return cls.model_validate(raw)


# --- Unity Catalog metadata -----------------------------------------------------------
class ColumnSummary(BaseModel):
    name: str
    type: str
    comment: str = ""


class TableSummary(BaseModel):
    name: str
    comment: str = ""
    columns: list[ColumnSummary] = Field(default_factory=list)


# --- Tool arguments (their JSON schemas become the LLM tool definitions) ---------------
class DescribeTableArgs(BaseModel):
    """Columns, types, comments and 3 sample rows of one table in the weather schema."""

    table_name: str = Field(description="Unqualified table name from the overview.")


class RunSqlArgs(BaseModel):
    """Run ONE read-only Databricks SQL SELECT/WITH statement. Returns a preview and a query_index."""

    query: str = Field(description="The SQL statement.")
    purpose: str = Field(max_length=200, description="One short sentence: what this query answers.")

    @field_validator("query")
    @classmethod
    def _read_only(cls, value: str) -> str:
        return validate_sql(value)  # UnsafeSQLError is a ValueError -> ValidationError


class ChartSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ChartKind
    x: str = Field(description="x-axis column (longitude for map).")
    y: str | None = Field(None, description="y-axis column (latitude for map). Optional for histogram.")
    color: str | None = Field(None, description="Optional column that splits the series.")
    title: str = Field(max_length=120, description="Short title with units.")

    @model_validator(mode="after")
    def _y_required(self) -> ChartSpec:
        if self.kind != "histogram" and not self.y:
            raise ValueError(f"'{self.kind}' charts need a y column")
        return self


class AgentOutput(BaseModel):
    """Final answer. Submit exactly once with submit_answer."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, description="Direct answer in 1-4 sentences, with units.")
    query_index: int | None = Field(
        None, ge=0, description="query_index of the run_sql result that supports the answer."
    )
    chart: ChartSpec | None = Field(None, description="Optional chart from the result at query_index.")
    follow_ups: list[str] = Field(default_factory=list, description="Short follow-up questions.")
    confidence: Literal["high", "medium", "low"]

    @model_validator(mode="after")
    def _matches_agent_yaml(self, info: ValidationInfo) -> AgentOutput:
        """Check the output against agent.yaml rules and the queries that actually ran.

        Context keys: rules (OutputRules), results ({query_index: {column: is_numeric}}).
        """
        context = info.context or {}
        rules: OutputRules | None = context.get("rules")
        results: dict[int, dict[str, bool]] = context.get("results", {})
        if rules is None:
            return self

        problems = []
        if len(self.answer) > rules.answer_max_chars:
            problems.append(f"answer is longer than {rules.answer_max_chars} characters")
        if len(self.follow_ups) > rules.max_follow_ups:
            problems.append(f"at most {rules.max_follow_ups} follow_ups are allowed")
        if self.query_index is not None and self.query_index not in results:
            problems.append(f"query_index {self.query_index} is not a successful run_sql result")
        if rules.require_query_for_numbers and not results and _MEASUREMENT.search(self.answer):
            problems.append("the answer states measurements but no run_sql query succeeded")

        if self.chart is not None:
            if self.chart.kind not in rules.chart_kinds:
                problems.append(f"chart kind must be one of {rules.chart_kinds}")
            if self.query_index is None:
                problems.append("a chart needs query_index")
            elif self.query_index in results:
                columns = results[self.query_index]
                for role in ("x", "y", "color"):
                    column = getattr(self.chart, role)
                    if column and column not in columns:
                        problems.append(f"chart {role}='{column}' is not in result columns {list(columns)}")
                if self.chart.kind == "map" and not all(columns.get(c, False) for c in (self.chart.x, self.chart.y)):
                    problems.append("map needs numeric longitude (x) and latitude (y) columns")

        if problems:
            raise ValueError("; ".join(problems))
        return self


# --- Results/What the app receives --------------------------------------------------------------
class QueryRecord(BaseModel):
    index: int
    sql: str
    purpose: str = ""
    ok: bool
    rows: int = 0
    error: str | None = None


class AgentResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: AgentOutput
    valid: bool = Field(description="True when the output passed validation.")
    repairs: int = 0
    queries: list[QueryRecord] = Field(default_factory=list)
    data: pd.DataFrame | None = None
    steps: list[str] = Field(default_factory=list)


def tool_spec(name: str, model: type[BaseModel]) -> dict[str, Any]:
    """OpenAI-style tool definition generated from a Pydantic model (single source of truth)."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (model.__doc__ or "").strip(),
            "parameters": _inline_refs(model.model_json_schema()),
        },
    }


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace $ref/$defs with inline schemas; some model endpoints reject $ref."""
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(dict(defs[node["$ref"].split("/")[-1]]))
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)
