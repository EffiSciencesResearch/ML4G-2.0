# Slack Message Scheduler

**What it does:** Automatically sends Slack messages at scheduled times, exports channel member lists, and manages scheduled messages.

## Quick Start

### 1. First-Time Setup (takes ~10 minutes)

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

### 2. How to Use It

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

## Tips & Troubleshooting

- **Times:** All times are in YOUR computer's local timezone
- **Messages send as YOU:** Not from a bot - they'll appear as your Slack profile
- **Future only:** You can only schedule future messages (up to 120 days ahead)
- **Private channels:** You must be a member to send messages there
- **Error: "Token not found":** Check that your `.env` file has `SLACK_USER_TOKEN='xoxp-...'`
