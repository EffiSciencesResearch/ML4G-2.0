# notebook_tools

Authoring helpers for ML4G workshop notebooks: keep notebooks clean, badged, formatted, and in sync between solution and exercise variants. This is the daily-driver toolbox for anyone writing or maintaining a workshop.

## Commands

Invoke with `uv run python -m meta.notebook_tools <command>`.

- `neat` — the canonical one-shot command: runs `clean` + `badge` + `fmt` on the given notebooks. **Use this by default.**
- `clean` — strip outputs, execution counts, and per-cell metadata.
- `badge` — add or update the "Open in Colab" badge on a workshop notebook.
- `sync` — regenerate exercise/variant notebooks (`_normal`, `_hard`, ...) from a solution notebook using `# Hide:` annotations.
- `show_links` — list every URL referenced in the repository.
- `check_links` — flag broken or misrouted links (e.g. Colab links to deleted files).
- `fix_typos` — interactively fix typos in a notebook via gpt-3.5 (requires `OPENAI_API_KEY`).
- `list_of_workshops_readme` — rewrite the workshops index in the top-level `readme.md`.

## Examples

```bash
uv run python -m meta.notebook_tools neat workshops/
uv run python -m meta.notebook_tools sync workshops/my_workshop/my_workshop.ipynb
uv run python -m meta.notebook_tools check_links --curl
```

## Troubleshooting

- `fix_typos` errors with auth issues: ensure `OPENAI_API_KEY` is set in your environment (or `.env`).
- `check_links` flags links you know are fine: rate-limited hosts (GitHub raw, Colab) sometimes return 429 — re-run, or trust the manual check.
