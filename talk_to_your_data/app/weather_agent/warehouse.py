"""SQL warehouse access (Statement Execution API) and Unity Catalog schema overview."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pandas as pd

from .pydantic_models import AgentConfig, ColumnSummary, TableSummary

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

_NUMERIC = {"BYTE", "SHORT", "INT", "INTEGER", "LONG", "BIGINT", "FLOAT", "DOUBLE", "DECIMAL"}
_TEMPORAL = {"DATE", "TIMESTAMP", "TIMESTAMP_NTZ"}


class QueryError(RuntimeError):
    """SQL failed or timed out on the warehouse."""


def to_dataframe(columns: list[tuple[str, str]], rows: list[list[Any]]) -> pd.DataFrame:
    """JSON_ARRAY results arrive as strings; convert them to proper dtypes."""
    seen: dict[str, int] = {}
    names = []
    for name, _ in columns:  # make duplicate column names unique (joins)
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    df = pd.DataFrame(rows or [], columns=names, dtype=object)
    for name, (_, type_name) in zip(names, columns, strict=True):
        kind = (type_name or "").upper()
        if kind in _NUMERIC:
            df[name] = pd.to_numeric(df[name], errors="coerce")
        elif kind in _TEMPORAL:
            df[name] = pd.to_datetime(df[name], errors="coerce", utc=True).dt.tz_localize(None)
        elif kind == "BOOLEAN":
            df[name] = df[name].map({"true": True, "false": False})
    return df


class WarehouseClient:
    def __init__(self, ws: WorkspaceClient, warehouse_id: str, config: AgentConfig) -> None:
        self._ws = ws
        self._warehouse_id = warehouse_id
        self._config = config

    def query(self, sql: str) -> pd.DataFrame:
        from databricks.sdk.service.sql import (
            Disposition,
            ExecuteStatementRequestOnWaitTimeout,
            Format,
            StatementState,
        )

        limits, api = self._config.limits, self._ws.statement_execution
        response = api.execute_statement(
            statement=sql,
            warehouse_id=self._warehouse_id,
            catalog=self._config.data.catalog,
            schema=self._config.data.schema_name,
            disposition=Disposition.INLINE,
            format=Format.JSON_ARRAY,
            row_limit=limits.max_rows,
            wait_timeout="30s",
            on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
        )
        deadline = time.monotonic() + limits.query_timeout_s
        while response.status and response.status.state in (StatementState.PENDING, StatementState.RUNNING):
            if time.monotonic() > deadline:
                api.cancel_execution(response.statement_id)
                raise QueryError(f"Query timed out after {limits.query_timeout_s}s.")
            time.sleep(1.5)
            response = api.get_statement(response.statement_id)

        if not response.status or response.status.state != StatementState.SUCCEEDED:
            error = response.status.error if response.status else None
            raise QueryError(error.message if error and error.message else "Query failed.")

        schema = response.manifest.schema if response.manifest else None
        columns = [
            (c.name, str(getattr(c.type_name, "value", c.type_name) or c.type_text))
            for c in (schema.columns if schema and schema.columns else [])
        ]
        rows: list[list[Any]] = list(response.result.data_array or []) if response.result else []
        next_chunk = response.result.next_chunk_index if response.result else None
        while next_chunk is not None and len(rows) < limits.max_rows:
            chunk = api.get_statement_result_chunk_n(response.statement_id, next_chunk)
            rows.extend(chunk.data_array or [])
            next_chunk = chunk.next_chunk_index
        return to_dataframe(columns, rows[: limits.max_rows])


def load_tables(ws: WorkspaceClient, catalog: str, schema: str) -> list[TableSummary]:
    tables = [
        TableSummary(
            name=t.name,
            comment=t.comment or "",
            columns=[
                ColumnSummary(name=c.name, type=c.type_text or "", comment=c.comment or "") for c in (t.columns or [])
            ],
        )
        for t in ws.tables.list(catalog_name=catalog, schema_name=schema)
    ]
    return sorted(tables, key=lambda t: t.name)


def render_overview(tables: list[TableSummary]) -> str:
    lines = []
    for table in tables:
        lines.append(f"\n### {table.name}" + (f" — {table.comment}" if table.comment else ""))
        lines += [f"- {c.name} ({c.type})" + (f" — {c.comment}" if c.comment else "") for c in table.columns]
    return "\n".join(lines).strip()
