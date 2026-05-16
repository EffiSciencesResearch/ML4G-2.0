# Feedback Form Creator

Creates the daily Google Forms feedback questionnaire for each day of a camp.
Two interfaces share the same form-builder code:

- **Web UI**: the **Feedback Form Creator** page of the [internal Streamlit app](../web/README.md). The YAML config lives on the camp object (`camp.feedback_config_yaml`); the page renders it in a code editor, validates it, and creates the form for the selected day.
- **CLI**: `uv run python -m meta.feedback_form_creator` — interactive prompt that picks a day from `meta/feedback_form_creator/config.yaml` and creates that day's form.

For most camps you'll use the web UI. The CLI is what we used historically and still works.

## Configuration

The config is YAML with the structure defined in [`models.py`](./models.py) (`CampConfig`). Highlights:

```yaml
camp_name: "ML4good UK 2025"          # appears in form titles
drive_folder_id: "1A..."              # forms get moved into this Drive folder
teachers: [Diego, Rich, Elsa, Joël]
form_description: |
  ...intro shown at the top of every form...
pre_questions:                        # asked at the top of every day's form
  - text: Name
    kind: choice
    choices: [Alice, Bob]
    mandatory: true
timetable:                            # one entry per camp day
  day_1:
    meme: drake.png                   # see ./memes/ for available images
    sessions:
      - name: Opening session
      - name: Chapter 1
        reading_group: true           # adds a "who facilitated?" question
    day_questions:                    # optional, day-specific extras
      - text: "Are you well settled?"
        kind: paragraph
post_questions:                       # asked at the end of every day's form
  - text: "How would you rate today?"
    kind: scale
```

For each `day_N`, the generated form contains: pre-questions → per-session rating + (reading-group teacher choice) + feedback → day-specific questions → post-questions → meme image + caption.

The exact list for any given day is shown by the **Preview** in the web UI; under the hood, `build_question_plan` in `cli.py` is the single source of truth.

### VS Code autocompletion

`config.schema.json` is wired up in `.vscode/settings.json`. Install the [Red Hat YAML extension](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml). Regenerate after editing `models.py`:

```bash
uv run python -m meta.feedback_form_creator.models
```

### Memes

Drop a `.png` or `.jpg` into [`memes/`](./memes/) and reference it by filename under `day_N.meme`. See [memes/README.md](./memes/) for the gallery of what's currently available.

## Authentication

Two auth paths, in order of preference:

1. **Service account** (used by the deployed web app and the recommended way locally): set `SERVICE_ACCOUNT_JSON` env var to the contents of a Google Cloud service-account key. The same key is used by other Drive-backed pages.
2. **OAuth2** (CLI-only fallback): drop `creds.json` from [Google Cloud Console](https://console.cloud.google.com/apis/credentials) (Desktop OAuth client) into this folder. On first run, a browser window opens; `token.pickle` is then cached for subsequent runs.

In both cases, the **Google Forms API** and **Google Drive API** must be enabled in the relevant Cloud project, and the bot must have **Editor** rights on the target `drive_folder_id`. The web UI surfaces a targeted "give Editor to `<bot>` on the folder" message on 403.

`creds.json` and `token.pickle` are gitignored — never commit them.

## File structure

```
meta/feedback_form_creator/
├── web.py            # Streamlit page (Feedback Form Creator)
├── cli.py            # CLI entry point + build_question_plan + create_daily_feedback_form
├── forms_utils.py    # Google Forms/Drive API wrappers
├── models.py         # Pydantic config models + schema generator
├── config.yaml       # Sample / CLI config
├── config.schema.json
└── memes/
    ├── README.md     # gallery of available images
    └── *.png, *.jpg
```

## Adding a new question kind

1. Add the Pydantic model in [`models.py`](./models.py) and include it in `AnyQuestionConfig`.
2. Add an `if config.kind == "..."` branch to `create_question_from_config()` in [`forms_utils.py`](./forms_utils.py).
3. Regenerate the schema (`uv run python -m meta.feedback_form_creator.models`).

The new kind is automatically picked up by both the web preview and the form creator via `build_question_plan`.
