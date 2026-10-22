import os
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
import pandas as pd
import pytest
from pydantic import ValidationError
from nlq.config import AgentConfig, ModelCfg
from nlq.genie import GenieClient
from nlq.viz import ChartSpec, make_figure

DF = pd.DataFrame({"region": ["EU", "US"], "revenue": [10.0, 20.0]})

def test_valid_spec_makes_chart():
    assert make_figure(ChartSpec(chart="bar", x="region", y="revenue"), DF) is not None

@pytest.mark.parametrize("spec", [
    ChartSpec(chart="bar", x="nope", y="revenue"),
    ChartSpec(chart="line", x="revenue", y="region"),
    ChartSpec(chart="table"),
])
def test_invalid_spec_falls_back_to_table(spec):
    assert make_figure(spec, DF) is None

def test_unresolved_env_var_rejected():
    with pytest.raises(ValidationError):
        ModelCfg(endpoint="${LLM}")

def test_genie_parses_result():
    ws = MagicMock()
    ws.genie.start_conversation_and_wait.return_value = NS(
        conversation_id="c", message_id="m",
        attachments=[NS(attachment_id="a", text=None, query=NS(query="SELECT 1"))])
    stmt = NS(manifest=NS(schema=NS(columns=[NS(name="region"), NS(name="revenue")])),
              result=NS(data_array=[["EU", "10"], ["US", "20"]]))
    ws.genie.get_message_attachment_query_result.return_value = NS(statement_response=stmt)
    r = GenieClient(ws, "space", max_rows=1).ask("q")
    assert r.sql == "SELECT 1" and len(r.df) == 1 and r.df.revenue.dtype.kind in "if"

@pytest.mark.skipif(not os.getenv("GENIE_SPACE_ID"), reason="needs a workspace")
def test_genie_live():
    from databricks.sdk import WorkspaceClient
    r = GenieClient(WorkspaceClient(), os.environ["GENIE_SPACE_ID"], 10).ask("How many tables are there?")
    assert r.sql
