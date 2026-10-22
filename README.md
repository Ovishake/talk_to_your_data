# talk_to_your_data

> Ask plain-English questions about your data — get an AI-written summary, an interactive chart, and the underlying SQL, all in one click.

![App screenshot](docs/screenshot.png)

---

## What it does

`talk_to_your_data` is a Streamlit app deployed on **Databricks Apps** that lets anyone query structured data through natural language. Type a question (or pick an example), and the app:

1. Forwards the question to **Databricks Genie** to generate and run the SQL.
2. Passes the result to a **pydantic-ai** agent backed by a Databricks-hosted LLM.
3. Returns a plain-English **summary**, an interactive **Plotly chart**, a paginated **data table**, and the raw **SQL** — all without writing a single line of code.

---

## Example queries

| Question | Chart type |
| --- | --- |
| Average maximum temperature per day | Line |
| The 5 days with the most precipitation | Bar |
| Minimum and maximum temperature per location | Bar |
| Average temperature of Tokyo and Berlin over the days | Line |

---


## Architecture

```
User (browser)
    │
    ▼
Streamlit UI  (app.py)
    │
    ├── pydantic-ai Agent  (nlq/agent.py)
    │       └── ask_genie tool
    │               │
    │               ▼
    │         Databricks Genie  ──▶  Unity Catalog table
    │               │
    │               └── SQL + rows returned
    │
    ├── LLM endpoint  (databricks-claude-haiku-5-5)
    │       └── summary + chart spec
    │
    └── Plotly chart  (nlq/viz.py)
```

---

## Project structure

```
talk_to_your_data/
├── app.py                  # Streamlit entry point
├── app.yaml                # Databricks Apps runtime config (env vars, command)
├── agent.yaml              # Agent definition: model, genie, data, guardrails, UI
├── databricks.yml          # DAB bundle config (dev + prod targets)
├── requirements.txt        # Runtime Python dependencies
├── pyproject.toml          # Project metadata + dev dependencies
├── nlq/
│   ├── agent.py            # pydantic-ai agent + ask_genie tool
│   ├── config.py           # Pydantic config models, env-var expansion
│   ├── genie.py            # Genie SDK wrapper → GenieResult
│   └── viz.py              # ChartSpec model + Plotly figure builder
├── resources/
│   └── talk_to_your_data.app.yml   # App resource definition
└── tests/
    └── test_core.py
```

---

## Prerequisites

- Databricks workspace on Azure with **Unity Catalog** enabled
- A **Genie Space** configured over your target table(s)
- A model serving endpoint (default: `databricks-claude-haiku-5-5`)
- Databricks CLI ≥ 0.230 with bundle support
- Python ≥ 3.11

---

## Configuration

All runtime values are set in `app.yaml` as plain `value:` entries (do **not** use `valueFrom:` — it is not supported in bundle-deployed apps):

```yaml
# app.yaml
command: ["streamlit", "run", "app.py"]
env:
  - name: GENIE_SPACE_ID
    value: <your-genie-space-id>
  - name: LLM_ENDPOINT
    value: databricks-claude-haiku-5-5
```

The agent behaviour (data catalog/schema, guardrails, UI copy, example queries, etc.) lives entirely in `agent.yaml` and can be changed without touching any Python code.

---

## Deploy

```bash
# 1. Validate
databricks bundle validate --strict -t dev

# 2. Deploy (source-linked — no file upload needed)
databricks bundle deploy -t dev

# 3. Start / restart the app
databricks bundle run talk_to_your_data -t dev
```

The app URL is printed at the end of `bundle run`:

```
You can access the app at https://ttyd-<username>-<workspace-id>.azure.databricksapps.com
```

For production, substitute `-t prod`.

---

## Tech stack

| Layer | Library |
| --- | --- |
| UI | Streamlit |
| Agent framework | pydantic-ai |
| LLM | Databricks model serving (OpenAI-compatible) |
| Data Q&A | Databricks Genie SDK |
| Charts | Plotly Express |
| Config | Pydantic v2 + PyYAML |
| Observability | MLflow autolog |
| Deployment | Declarative Automation Bundles (DABs) + Databricks Apps |

---

## Guardrails

Blocked SQL keywords (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `MERGE`, `ALTER`, `TRUNCATE`, `GRANT`) and hidden columns (`email`, `phone`, `iban`) are defined in `agent.yaml` under `guardrails:` and enforced at the agent layer before any result reaches the UI.

---

## License

See [LICENSE](LICENSE).
