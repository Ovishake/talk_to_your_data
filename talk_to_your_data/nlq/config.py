import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

NO_PLACEHOLDER = r"^[^$]+$"  # rejects unresolved ${VARS}
ChartType = Literal["bar", "line", "scatter", "pie", "table"]


class ModelCfg(BaseModel):
    endpoint: str = Field(pattern=NO_PLACEHOLDER)
    temperature: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int = Field(default=1024, gt=0)
    timeout_s: float = Field(default=60, gt=0)
    retries: int = Field(default=2, ge=0, le=5)


class GenieCfg(BaseModel):
    space_id: str = Field(pattern=NO_PLACEHOLDER)
    max_rows: int = Field(default=5000, gt=0, le=100_000)


class Table(BaseModel):
    name: str
    description: str = ""


class DataCfg(BaseModel):
    catalog: str
    schema_: str = Field(alias="schema")
    tables: list[Table] = []
    glossary: dict[str, str] = {}


class Guardrails(BaseModel):
    blocked_sql_keywords: list[str] = []
    hidden_columns: list[str] = []


class OutputCfg(BaseModel):
    chart_types: list[ChartType] = ["bar", "line", "scatter", "pie", "table"]
    summary_max_sentences: int = Field(default=3, gt=0, le=10)


class UiCfg(BaseModel):
    title: str = "Ask the silver layer"
    placeholder: str = "Ask a question about the data"
    page_size: int = Field(default=100, gt=0)
    examples: list[str] = []


class CacheCfg(BaseModel):
    ttl_s: int = Field(default=300, ge=0)


class ObsCfg(BaseModel):
    mlflow_experiment: str | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class AgentConfig(BaseModel):
    name: str
    version: str
    description: str = ""
    owner: str = ""
    model: ModelCfg
    genie: GenieCfg
    data: DataCfg
    guardrails: Guardrails = Guardrails()
    output: OutputCfg = OutputCfg()
    ui: UiCfg = UiCfg()
    cache: CacheCfg = CacheCfg()
    observability: ObsCfg = ObsCfg()
    instructions: str

    def prompt(self) -> str:
        d = self.data
        tables = "\n".join(
            f"- {d.catalog}.{d.schema_}.{t.name}: {t.description}" for t in d.tables
        )
        terms = "\n".join(f"- {k}: {v}" for k, v in d.glossary.items())
        return (
            f"{self.instructions}\nAllowed chart types: {', '.join(self.output.chart_types)}.\n"
            f"Summary: max {self.output.summary_max_sentences} sentences.\n"
            f"Tables:\n{tables}\nGlossary:\n{terms}"
        )


def load_config(path: str | Path = "agent.yaml") -> AgentConfig:
    return AgentConfig.model_validate(
        yaml.safe_load(os.path.expandvars(Path(path).read_text()))
    )
