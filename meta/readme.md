# ML4G Automation Tools - User Guide

This folder contains helpful scripts to automate common tasks. No programming experience needed!

---

## Table of Contents
1. [Slack Message Scheduler](./slack_reminders/README.md) - Schedule Slack messages automatically
2. [Google Docs Duplicator](#google-docs-duplicator) - Create personalized Google Docs for everyone
3. [Bootcamp Template Duplicator](#bootcamp-template-duplicator) - One-click duplicate the whole template tree for a new camp

---

## Google Docs Duplicator

**What it does:** Creates personalized copies of a Google Doc template for multiple people.

### What You Need

Before running this script, prepare:

1. **A Google Drive folder** - Where the new documents will be stored
2. **A Google Docs template** - The document you want to copy for everyone
3. **A CSV file** - List of people (must have columns: `email` and `name`)

   Example `names_and_emails.csv`:
   ```
   email,name
   alice@example.com,Alice Smith
   bob@example.com,Bob Jones
   ```

4. **A service account token** - This is a special file that lets the script access Google Drive
   - Ask Diego for a `service_account_token.json` file, OR
   - Get one from the Google Cloud Console (advanced users only)
   - Give the service account:
     - Write access to your Drive folder
     - View access to your template document

### How to Run It

Open your terminal in the ML4G2.0 folder and run:

```shell
uv run python -m meta.googletools duplicate-career-docs names_and_emails.csv template_url drive_folder_url
```

Replace:
- `names_and_emails.csv` - Path to your CSV file
- `template_url` - The Google Docs URL of your template
- `drive_folder_url` - The Google Drive folder URL where docs should be created

**Example:**
```shell
uv run python -m meta.googletools duplicate-career-docs ./data/students.csv \
  "https://docs.google.com/document/d/abc123..." \
  "https://drive.google.com/drive/folders/xyz789..."
```

The script will create a personalized copy of the template for each person in your CSV and save it to the folder.

---

## Bootcamp Template Duplicator

**What it does:** Duplicates the whole bootcamp Drive template tree (folders,
docs, slides, sheets, forms) in one click, with `{{VAR}}` substitution in
titles and bodies, link rewriting, and permission replay. Designed to replace
the manual ~20-times-a-year cloning ritual.

Unlike the Python "Google Docs Duplicator" above (which makes per-person
copies of a single doc), this one duplicates a whole tree for a new camp.

### How to use it

Open the **New camp template duplicator** Google Sheet, then:

1. Fill in the **Template Variables** table — at minimum `LONG_NAME` and
   `_FOLDER_OR_DOC_TO_DUPLICATE` (which points to the template root folder).
2. Menu: **Template Duplicator → Duplicate template…**
3. First time only, grant the OAuth scopes it asks for.
4. Wait for the toast notifications. The final dialog gives you a link to
   the new folder and flags any unknown `{{NAME}}` placeholders it found.

The new tree is placed as a sibling of the duplicator spreadsheet itself, so
keep that spreadsheet inside the year's parent folder.

### When something looks off

- Open the **Run log** sheet inside the duplicator spreadsheet — every error,
  retry, and unknown-placeholder gets a row.
- For a fresh diagnostic, the menu has `Debug: chip URIs` and
  `Debug: dump variables`.

### Developer notes

Source lives in [`meta/template_duplicator/`](./template_duplicator/) and is
deployed via `clasp push -f`. See its [README](./template_duplicator/README.md)
for the file layout and how the substitution pipeline works.

---

## Getting Help

- Check the full documentation in each Python file for advanced options
- For Slack message scheduling, see [slack_reminders/README.md](./slack_reminders/README.md)
- Ask Diego if you get stuck!
