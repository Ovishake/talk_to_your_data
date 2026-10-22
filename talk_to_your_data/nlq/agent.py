import re
from dataclasses import dataclass
from databricks.sdk import WorkspaceClient
from openai import AsyncOpenAI
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from .config import AgentConfig
from .genie import GenieClient, GenieResult
from .viz import Answer

@dataclass
class Deps:
    genie: GenieClient
    result: GenieResult | None = None

def build_agent(cfg: AgentConfig, ws: WorkspaceClient) -> Agent[Deps, Answer]:
    token = ws.config.authenticate()["Authorization"].removeprefix("Bearer ")
    client = AsyncOpenAI(base_url=f"{ws.config.host}/serving-endpoints", api_key=token)
    m = cfg.model
    agent = Agent(
        OpenAIChatModel(m.endpoint, provider=OpenAIProvider(openai_client=client)),
        deps_type=Deps, output_type=Answer, name=cfg.name,
        instructions=cfg.prompt(), retries=m.retries,
        model_settings=ModelSettings(**({"temperature": m.temperature} if m.temperature is not None else {}), max_tokens=m.max_tokens, timeout=m.timeout_s))
    kws = cfg.guardrails.blocked_sql_keywords
    blocked = re.compile(rf"\b({'|'.join(map(re.escape, kws))})\b", re.I) if kws else None

    @agent.tool
    def ask_genie(ctx: RunContext[Deps], question: str) -> str:
        """Ask Databricks Genie a data question about the silver tables."""
        r = ctx.deps.genie.ask(question)
        if blocked and r.sql and blocked.search(r.sql):
            raise ModelRetry("Only read-only SELECT queries are allowed.")
        r.df = r.df.drop(columns=cfg.guardrails.hidden_columns, errors="ignore")
        ctx.deps.result = r
        return (f"SQL: {r.sql}\nGenie: {r.text}\nRows: {len(r.df)}\n"
                f"Dtypes: {r.df.dtypes.astype(str).to_dict()}\n"
                f"Sample:\n{r.df.head(10).to_csv(index=False)}")

    @agent.output_validator
    def check_output(ctx: RunContext[Deps], out: Answer) -> Answer:
        if ctx.deps.result is None:
            raise ModelRetry("Call ask_genie first.")
        if out.spec.chart not in cfg.output.chart_types:
            raise ModelRetry(f"Use one chart type from: {cfg.output.chart_types}")
        cols = set(ctx.deps.result.df.columns)
        if cols and (bad := {c for c in (out.spec.x, out.spec.y, out.spec.color) if c} - cols):
            raise ModelRetry(f"Unknown columns {bad}. Use only: {sorted(cols)}")
        return out

    return agent
