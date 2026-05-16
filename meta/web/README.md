# `meta/web` — internal Streamlit dashboard

Internal Streamlit dashboard for ML4G bootcamp operations: camp creation and editing, career-docs duplication, session-feedback analysis, and one-on-one scheduling. Everything in this folder is used only by the web app; shared helpers (e.g. Google APIs) live in `meta/shared/`.

## Run locally

```
make run
# or:
uv run streamlit run meta/web/main.py --server.port 8991
```

## Deployment

Deployed to Fly.io via `fly.toml` and `Dockerfile` at the repo root.
