# ML4G Automation Tools - User Guide

This folder contains helpful scripts to automate common tasks. No programming experience needed!

---

## Table of Contents
1. [Slack Message Scheduler](#slack-message-scheduler) - Schedule Slack messages automatically
2. [Google Docs Duplicator](#google-docs-duplicator) - Create personalized Google Docs for everyone

---

## Slack Message Scheduler

**What it does:** Automatically sends Slack messages at scheduled times, exports channel member lists, and manages scheduled messages.

### Quick Start

#### 1. First-Time Setup (takes ~10 minutes)

**Get your Slack token:**
1. Go to https://api.slack.com/apps
2. Click "Create New App" → "From scratch"
3. Name it (e.g., "Message Scheduler") and select your workspace
4. Click "OAuth & Permissions" in the left sidebar
5. Under "User Token Scopes", add these 6 permissions:
   - `chat:write`
   - `users:read`
   - `users:read.email`
   - `channels:read`
   - `groups:read`
   - `im:write`
6. Click "Install to Workspace" at the top → Allow
7. Copy the "User OAuth Token" (starts with `xoxp-`)

**Save your token:**
1. In the ML4G2.0 folder, create a file called `.env` (if it doesn't exist)
2. Add this line (replace with your actual token):
   ```
   SLACK_USER_TOKEN='xoxp-your-very-long-token-here'
   ```
3. Save the file

**Test it works:**
```shell
uv run python -m meta.slack_reminders --help
```

If you see help text, you're ready to go!

#### 2. How to Use It

**Schedule messages from a spreadsheet (CSV):**

1. Create a spreadsheet with these exact columns:
   - `content` - The message text
   - `date` - When to send (format: `2025-10-05 14:30`). If you want to send them now, a good practice is to schedule them for a bit later, in case you realise you messed up something.
   - `destination` - Who gets it (`@alice`, `alice@example.com`, or `#channel-name`)

2. Save as CSV (e.g., `messages.csv`)

3. Test first (safe, doesn't send, but says what will be sent):
   ```shell
   uv run python -m meta.slack_reminders schedule-csv /path/to/messages.csv --dry-run
   ```

4. If it looks good, send for real:
   ```shell
   uv run python -m meta.slack_reminders schedule-csv /path/to/messages.csv
   ```

**Schedule from a template (YAML):**

Great for sending personalized messages to multiple people. Create a YAML file like this:

```yaml
date: '2025-10-05 14:30'
template: 'Hi {name}, your session is on {topic}!'
variables:
  name:
    alice: Alice
    bob: Bob
  topic:
    alice: Python
    bob: Machine Learning
user_id:
  alice: U12345678
  bob: U87654321
```

Then run:
```shell
uv run python -m meta.slack_reminders schedule-yaml /path/to/reminders.yaml
```

**Export channel members:**

Get a list of everyone in a channel with their emails:
```shell
uv run python -m meta.slack_reminders export-channel-members --output-csv members.csv
```

**Delete scheduled messages:**

Changed your mind about scheduled messages?
```shell
uv run python -m meta.slack_reminders delete-scheduled
```
This will show you the list of scheduled messages and allow you to unschedule some of them. It's quite convenient because they don't show up in slack as scheduled.

### Tips & Troubleshooting

- **Times:** All times are in YOUR computer's local timezone
- **Messages send as YOU:** Not from a bot - they'll appear as your Slack profile
- **Future only:** You can only schedule future messages (up to 120 days ahead)
- **Private channels:** You must be a member to send messages there
- **Error: "Token not found":** Check that your `.env` file has `SLACK_USER_TOKEN='xoxp-...'`

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

## Getting Help

- Check the full documentation in each Python file for advanced options
- For `slack_reminders.py`, run: `uv run python -m meta.slack_reminders --help`
- Ask Diego if you get stuck!
