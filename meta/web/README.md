# Streamlit Dashboard

Internal Streamlit dashboard for ML4G bootcamp operations: camp creation and editing, career-docs duplication, session-feedback analysis, and one-on-one scheduling.

The dashboard is composed of pages from across the `meta/` tree. App-frame pages (dashboard, create_camp, edit_camp, session_feedback) live in `meta/web/pages/`. Pages that belong to a specific tool live in that tool's folder as `web.py` (`meta/one_on_ones/web.py`, `meta/career_docs/web.py`). The page registry is in `meta/web/main.py`; domain models and service wrappers live in `meta/shared/`.

## Prerequisites

- `uv sync` to install dependencies.
- A `.env` at the repo root with (as needed by the pages you use):
  - `ML4G_PORTAL_ADMIN_PASSWORD` — gates the camp-creation page.
  - `OPENROUTER_PROVISIONING_KEY` — used by the edit-camp page for LLM calls.
  - `S3_BUCKET_NAME` (+ standard AWS creds) — optional, enables persistent storage; falls back to local storage if unset.
- `service_account_token.json` at the repo root for the Google-Drive-backed pages (career docs, session feedback).

## Run locally

```shell
make run
# or
uv run streamlit run meta/web/main.py --server.port 8991
```

## Deployment

Deployed to Fly.io via `fly.toml` and `Dockerfile` at the repo root.

## Troubleshooting

- A page errors on Google API calls: check `service_account_token.json` is present and has access to the relevant Drive resources.
- Session state resets unexpectedly: without `S3_BUCKET_NAME`, state is local to the running process and lost on restart.
