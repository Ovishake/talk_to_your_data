import logging
from dataclasses import dataclass
import pandas as pd
from databricks.sdk import WorkspaceClient

log = logging.getLogger(__name__)

@dataclass
class GenieResult:
    sql: str | None
    text: str | None
    df: pd.DataFrame

class GenieClient:
    def __init__(self, ws: WorkspaceClient, space_id: str, max_rows: int):
        self.ws, self.space_id, self.max_rows = ws, space_id, max_rows

    def ask(self, question: str) -> GenieResult:
        msg = self.ws.genie.start_conversation_and_wait(self.space_id, question)
        sql, text, df = None, None, pd.DataFrame()
        for att in msg.attachments or []:
            if att.text:
                text = att.text.content
            if att.query:
                sql = att.query.query
                res = self.ws.genie.get_message_attachment_query_result(
                    self.space_id, msg.conversation_id, msg.message_id, att.attachment_id)
                df = to_frame(res.statement_response, self.max_rows)
        log.info("genie_answer rows=%d sql=%r", len(df), sql)
        return GenieResult(sql, text, df)

def to_frame(stmt, max_rows: int) -> pd.DataFrame:
    cols = [c.name for c in stmt.manifest.schema.columns]
    rows = (stmt.result.data_array or [])[:max_rows] if stmt.result else []
    df = pd.DataFrame(rows, columns=cols)
    for c in df.columns:  # Genie returns strings; convert fully numeric columns
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().sum() == df[c].notna().sum():
            df[c] = num
    return df
