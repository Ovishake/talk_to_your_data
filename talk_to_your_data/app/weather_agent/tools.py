"""Tool execution. Arguments are validated with the Pydantic models in pydantic_models.py."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import ValidationError

from .pydantic_models import (
    AgentConfig,
    AgentOutput,
    DescribeTableArgs,
    QueryRecord,
    RunSqlArgs,
    TableSummary,
    tool_spec,
)
from .warehouse import QueryError

if TYPE_CHECKING:
    from .warehouse import WarehouseClient

MAX_TOOL_OUTPUT_CHARS = 8000
TOOL_MODELS = {"describe_table": DescribeTableArgs, "run_sql": RunSqlArgs, "submit_answer": AgentOutput}


def tool_specs(config: AgentConfig) -> list[dict[str, Any]]:
    return [tool_spec(name, TOOL_MODELS[name]) for name in config.tools]


def to_json(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, default=str, ensure_ascii=False)
    return text if len(text) <= MAX_TOOL_OUTPUT_CHARS else text[:MAX_TOOL_OUTPUT_CHARS] + "…[truncated]"


def errors_text(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'output'}: {e['msg']}" for e in exc.errors())


def _preview(df: pd.DataFrame, rows: int) -> list[dict[str, Any]]:
    return json.loads(df.head(rows).to_json(orient="records", date_format="iso"))


class ToolBox:
    """Per-question state. Create one per question, so concurrent users never share state."""

    def __init__(self, warehouse: WarehouseClient, tables: list[TableSummary], config: AgentConfig) -> None:
        self._warehouse = warehouse
        self._tables = {t.name.lower(): t for t in tables}
        self._config = config
        self.queries: list[QueryRecord] = []
        self.results: dict[int, pd.DataFrame] = {}

    def output_context(self) -> dict[str, Any]:
        """Validation context for AgentOutput: rules from agent.yaml + columns of each result."""
        return {
            "rules": self._config.output,
            "results": {
                i: {c: pd.api.types.is_numeric_dtype(df[c]) for c in df.columns} for i, df in self.results.items()
            },
        }

    def run(self, name: str, raw_arguments: str | None) -> str:
        if name not in ("describe_table", "run_sql"):
            return to_json({"error": f"Unknown tool '{name}'."})
        try:
            if name == "describe_table":
                return self._describe_table(DescribeTableArgs.model_validate_json(raw_arguments or "{}"))
            return self._run_sql(RunSqlArgs.model_validate_json(raw_arguments or "{}"))
        except ValidationError as exc:
            if name == "run_sql":
                self._record_rejected(raw_arguments, errors_text(exc))
            return to_json({"error": f"Invalid {name} arguments: {errors_text(exc)}"})

    def _record_rejected(self, raw_arguments: str | None, error: str) -> None:
        try:
            args = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        self.queries.append(QueryRecord(index=len(self.queries), sql=str(args.get("query", "")), ok=False, error=error))

    def _describe_table(self, args: DescribeTableArgs) -> str:
        table = self._tables.get(args.table_name.split(".")[-1].strip("`").lower())
        if table is None:
            return to_json({"error": f"Unknown table '{args.table_name}'.", "tables": sorted(self._tables)})
        payload: dict[str, Any] = table.model_dump()
        try:
            payload["sample_rows"] = _preview(self._warehouse.query(f"SELECT * FROM `{table.name}` LIMIT 3"), 3)
        except QueryError as exc:
            payload["sample_error"] = str(exc)
        return to_json(payload)

    def _run_sql(self, args: RunSqlArgs) -> str:
        index = len(self.queries)
        try:
            df = self._warehouse.query(args.query)
        except QueryError as exc:
            self.queries.append(
                QueryRecord(index=index, sql=args.query, purpose=args.purpose, ok=False, error=str(exc))
            )
            return to_json({"error": f"Query failed: {exc}"})

        self.results[index] = df
        self.queries.append(QueryRecord(index=index, sql=args.query, purpose=args.purpose, ok=True, rows=len(df)))
        limits = self._config.limits
        payload: dict[str, Any] = {
            "query_index": index,
            "row_count": len(df),
            "columns": {c: str(t) for c, t in df.dtypes.items()},
            "preview": _preview(df, limits.preview_rows),
        }
        if len(df) >= limits.max_rows:
            payload["note"] = f"Result capped at {limits.max_rows} rows; aggregate further."
        return to_json(payload)
