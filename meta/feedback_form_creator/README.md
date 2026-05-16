# Automated Feedback Form Generator

This tool automatically creates daily feedback forms for ML4Good camps using the Google Forms API.

## Features

- 📝 Generates structured feedback forms for each day of the camp
- 🎯 Includes pre-questions, session ratings, day-specific questions, and post-questions
- 🎨 Adds fun memes to each form for engagement
- 📁 Organizes forms in a Google Drive folder
- 🔐 Secure authentication with credential caching

## Prerequisites

- Python 3.11 (as specified in `pyproject.toml`)
- Google account with access to Google Forms and Google Drive
- `uv` for dependency management

## Setup Instructions

### 1. Install Dependencies

```bash
# Install all dependencies
uv sync

# Activate the environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

### 2. Set Up Google OAuth Credentials

You need to create OAuth credentials to allow the script to access Google Forms and Drive APIs:

1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Create a new project or select an existing one
3. Enable the following APIs:
   - Google Forms API
   - Google Drive API
4. Create credentials:
   - Click "Create Credentials" → "OAuth client ID"
   - Choose "Desktop app" as the application type
   - Give it a name (e.g., "ML4G Feedback Forms")
   - Download the JSON file
5. Save the downloaded file as `meta/feedback_form_creator/creds.json`

### 3. First-Time Authentication

When you run the script for the first time:

1. It will open your browser automatically
2. Log in with your Google account
3. Grant the requested permissions (Forms and Drive access)
4. The browser will show "The authentication flow has completed"
5. A `token.pickle` file will be created to store your credentials for future use

## Configuration

Edit `meta/feedback_form_creator/config.yaml` to customize for your camp:

### Essential Settings to Change

```yaml
# Camp name - appears in form titles
camp_name: "ML4good Italy 2025"

# Google Drive folder ID where form shortcuts will be created
# Get this from the folder URL: https://drive.google.com/drive/folders/[FOLDER_ID]
drive_folder_id: "your-folder-id-here"

# List of teachers/TAs
teachers:
  - "Diego"
  - "Julian"
  - "T-bo"
  - "Linda"
```

### Customizing the Schedule

Each day in the timetable has:
- `meme`: Fun image for the day (must exist in `memes/` folder)
- `sessions`: List of workshop/session names
- `day_questions` (optional): Extra questions specific to that day

Example day configuration:
```yaml
day_1:
  meme: "doge.png"
  sessions:
    - name: "Intro to AI Safety"
    - name: "Chapter 1 Capabilities"
      reading_group: true  # Adds teacher selection question
    - name: "Agents Workshop"
  day_questions:  # Optional day-specific questions
    - text: "What was your favorite part of day 1?"
      kind: "paragraph"
      mandatory: false
```

### Editor Integration (VSCode)

To get autocompletion and validation for `config.yaml` in VSCode:

1.  **Install the YAML extension**: Install the [YAML extension by Red Hat](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml) from the VSCode Marketplace.
2.  **Reload VSCode**: The schema is already configured in `.vscode/settings.json`.

To regenerate the schema after changing `models.py`, run:
```bash
uv run python -m meta.feedback_form_creator.models
```


## Usage

Run the script (from the repo root):
```bash
uv run python -m meta.feedback_form_creator
```

The script will:
1. Show available days from your config
2. Ask which day to create
3. Generate the form with all configured questions
4. Upload any meme images to Drive
5. Create a shortcut in your specified Drive folder
6. Display the form URLs

## File Structure

```
meta/feedback_form_creator/
├── cli.py           # Main script (entry point for `python -m meta.feedback_form_creator`)
├── forms_utils.py   # Helper functions
├── models.py        # Pydantic config models + schema generator
├── config.yaml      # Configuration file
├── config.schema.json  # Generated schema for editor autocompletion
├── creds.json       # OAuth credentials (git-ignored)
├── token.pickle     # Cached auth token (git-ignored)
└── memes/           # Meme images for each day
    ├── doge.png
    ├── drake.png
    └── ...
```

## Security Notes

- `creds.json` and `token.pickle` are automatically git-ignored
- Never commit these files to version control
- Each user needs their own OAuth credentials

## Troubleshooting

- **"Missing Google OAuth credentials"**: You need to download `creds.json` from Google Cloud Console
- **"The authentication flow has completed"**: Close the browser tab and check the terminal
- **Meme not found**: Ensure the meme filename in config.yaml exists in the `memes/` folder
- **Permission denied**: Make sure you enabled both Forms and Drive APIs in Google Cloud Console

## Adding New Features

To add new question types or features:
1. Add the question creation function in `forms_utils.py`
2. Update `create_question_from_config()` to handle the new type
3. Document the new type in this README
