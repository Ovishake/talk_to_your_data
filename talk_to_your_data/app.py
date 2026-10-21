import logging
import mlflow
import mlflow.pydantic_ai
import streamlit as st
from databricks.sdk import WorkspaceClient
from nlq.agent import Deps, build_agent
from nlq.config import load_config
from nlq.genie import GenieClient
from nlq.viz import make_figure

logging.basicConfig(format='{"t":"%(asctime)s","lvl":"%(levelname)s","log":"%(name)s","msg":"%(message)s"}')
log = logging.getLogger("nlq.app")

@st.cache_resource
def setup():
    cfg = load_config()
    logging.getLogger().setLevel(cfg.observability.log_level)
    if exp := cfg.observability.mlflow_experiment:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(exp)
        mlflow.pydantic_ai.autolog()
    log.info("started agent=%s version=%s", cfg.name, cfg.version)
    return cfg, WorkspaceClient()

cfg, ws = setup()

@st.cache_data(ttl=cfg.cache.ttl_s, show_spinner=False)
def ask(question: str):
    deps = Deps(GenieClient(ws, cfg.genie.space_id, cfg.genie.max_rows))
    return build_agent(cfg, ws).run_sync(question, deps=deps).output, deps.result

st.set_page_config(page_title=cfg.ui.title, layout="wide")
st.title(cfg.ui.title)
st.caption(cfg.description)
for col, ex in zip(st.columns(max(1, len(cfg.ui.examples))), cfg.ui.examples):
    if col.button(ex):
        st.session_state.q = ex
if q := st.chat_input(cfg.ui.placeholder):
    st.session_state.q = q
if q := st.session_state.get("q"):
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Asking Genie..."):
                ans, res = ask(q)
        except Exception:
            log.exception("query_failed")
            st.error("The query failed. Rephrase the question and try again.")
            st.stop()
        st.write(ans.summary)
        if res is None or res.df.empty:
            st.info("No data returned.")
            st.stop()
        if fig := make_figure(ans.spec, res.df):
            st.plotly_chart(fig)
        size = cfg.ui.page_size
        page = st.number_input("Page", 1, max(1, -(-len(res.df) // size)), 1)
        st.dataframe(res.df.iloc[(page - 1) * size: page * size])
        with st.expander("SQL"):
            st.code(res.sql or "", language="sql")
