# Google Ops

A small CLI of one-off Google Drive operations we run during bootcamps.

```shell
uv run python -m meta.google_ops --help
```

## Commands

### `duplicate-career-docs`

Creates personalized copies of a Google Doc template for multiple people.

**What you need:**

1. **A Google Drive folder** — where the new documents will be stored.
2. **A Google Docs template** — the document to copy for everyone. Use `[NAME]` as a placeholder; it will be replaced with each person's name in both the document body and the filename.
3. **A CSV file** with columns `email` and `name`:
   ```
   email,name
   alice@example.com,Alice Smith
   bob@example.com,Bob Jones
   ```
4. **A service account token** (`service_account_token.json` in the project root). Ask Diego, or create one in Google Cloud Console and give it:
   - Write access to your Drive folder
   - View access to your template document

**Run it:**

```shell
uv run python -m meta.google_ops duplicate-career-docs \
  ./data/students.csv \
  "https://docs.google.com/document/d/abc123..." \
  "https://drive.google.com/drive/folders/xyz789..."
```

A personalized copy is created for each person and shared with them.

### `copy-to-camp-folder`

Copy a Google Slides/Docs/Sheets file to a specified Drive folder, optionally with a new name prefix. The destination folder and prefix are cached in `config.yaml` so subsequent runs just need the URL.

```shell
uv run python -m meta.google_ops copy-to-camp-folder <slides-url> \
  --folder-url <drive-folder> \
  --camp-prefix "ML4G - "
```

### `add-prefix-to-folders`

Add a prefix to every folder name inside a given Drive folder. Has a `--dry-run` flag.

```shell
uv run python -m meta.google_ops add-prefix-to-folders <drive-folder-url> "ML4G - " --dry-run
```

## Config

`config.yaml` (created in this folder on first use) caches the destination folder URL and camp prefix for `copy-to-camp-folder`. It's gitignored — see project root `.gitignore`.
