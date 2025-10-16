"""Slack reminders CLI

Overview
========
This module provides two commands:
- schedule-csv: schedule Slack messages from a CSV so they send as YOU (user token)
- export-channel-members: fuzzy-select a channel and export members (id/name/email)

Obtaining a user token (xoxp-)
------------------------------
1) Create an app at `https://api.slack.com/apps` → Create New App → From scratch.
2) In the app, go to OAuth & Permissions → User Token Scopes and add:
   - chat:write
   - users:read
   - users:read.email
   - channels:read
   - groups:read
   - im:write
3) Click Install to Workspace (or Reinstall if you changed scopes). Approve the consent.
4) After install, copy the User OAuth Token (starts with xoxp-). Ignore any xoxb- token.
5) Add it to .env so the CLI can read it:
   export SLACK_USER_TOKEN='xoxp-...'

Verification
------------
Run a quick auth test:
  uv run python - <<'PY'\nimport os\nfrom slack_sdk import WebClient\nprint(WebClient(token=os.environ['SLACK_USER_TOKEN']).auth_test())\nPY

Usage examples
--------------
Schedule from CSV (dry run first):
  uv run python -m meta.slack_reminders schedule-csv /abs/path/messages.csv --dry-run
Schedule for real:
  uv run python -m meta.slack_reminders schedule-csv /abs/path/messages.csv

Export channel members with fuzzy search:
  uv run python -m meta.slack_reminders export-channel-members
Write members to CSV:
  uv run python -m meta.slack_reminders export-channel-members --output-csv /abs/path/members.csv

Notes
-----
- User tokens (xoxp-) post as you. Bot tokens (xoxb-) post as the app.
- Slack can schedule messages only for future times and up to ~120 days ahead.
- You must be a member to see private channels and their members.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional
import csv
import os
from datetime import datetime
import yaml

from dotenv import load_dotenv
import typer
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from InquirerPy import inquirer
from pydantic import BaseModel, ValidationError

load_dotenv()


app = typer.Typer(
    help="Schedule Slack messages from a CSV and export channel members.\n\n"
    "Auth: provide a USER OAuth token (xoxp-...) in SLACK_USER_TOKEN.\n\n"
    "CSV headers for scheduling: content,date,destination\n"
    "- content: message text\n"
    "- date: 'YYYY-MM-DD HH:MM' (interpreted in --tz)\n"
    "- destination: one of @username, email@example.com, Uxxxxxxxx, Dxxxxxxxx, Cxxxxxxxx, #channel_name\n\n"
    "Notes:\n"
    "- Using a user token posts as YOU. A bot token will post as the app.\n"
    "- Slack can schedule up to 120 days ahead; max ~30 scheduled messages per channel.\n\n"
    "Minimal user scopes: chat:write, users:read, users:read.email, channels:read, groups:read, im:write".replace(
        "\n", "\n\n"
    ),
    no_args_is_help=True,
    add_completion=False,
)


@dataclass
class MessageJob:
    content: str
    date_text: str
    destination: str
    post_at_epoch: int | None = None
    channel_id: str | None = None


def load_user_token() -> str:
    env_variable_name = "SLACK_USER_TOKEN"
    token = os.getenv(env_variable_name)
    if not token:
        raise RuntimeError(
            f"Environment variable {env_variable_name} is required and must contain a USER OAuth token (xoxp-...)."
        )
    if not token.startswith("xoxp-"):
        # Strong nudge: user asked to send as themselves; bot tokens won't.
        raise RuntimeError(
            f"{env_variable_name} must be a user token (xoxp-...). Got something else."
        )
    return token


def parse_csv(csv_path: Path) -> list[MessageJob]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"content", "date", "destination"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(
                f"CSV must have exact headers: {','.join(sorted(expected))}. Got: {reader.fieldnames}"
            )
        jobs: list[MessageJob] = []
        for row in reader:
            content = (row["content"] or "").strip()
            date_text = (row["date"] or "").strip()
            destination = (row["destination"] or "").strip()
            if not content or not date_text or not destination:
                raise ValueError(f"Row has empty fields: {row}")
            jobs.append(MessageJob(content=content, date_text=date_text, destination=destination))
        return jobs


def to_epoch_seconds(dt_text: str) -> int:
    dt = datetime.strptime(dt_text, "%Y-%m-%d %H:%M")
    return int(dt.timestamp())


def normalize_destination_value(value: str) -> str:
    # Keep simple and explicit
    if value.startswith("@"):
        return value[1:]
    if value.startswith("#"):
        return value[1:]
    return value


def find_or_open_channel_id(
    client: WebClient,
    destination: str,
) -> str:
    value = normalize_destination_value(destination)

    # Direct channel id provided
    if value.startswith("D"):
        return value
    # Channel id (public/private)
    if value.startswith("C") or value.startswith("G"):
        return value
    # User id
    if value.startswith("U") or value.startswith("W"):
        resp = client.conversations_open(users=[value])
        return resp["channel"]["id"]
    # Looks like an email
    if "@" in value and "." in value:
        user = client.users_lookupByEmail(email=value)["user"]["id"]
        resp = client.conversations_open(users=[user])
        return resp["channel"]["id"]

    # Treat as channel name first, then as username
    # Try public/private channel by name
    cursor: Optional[str] = None
    while True:
        convs = client.conversations_list(
            types="public_channel,private_channel",
            limit=1000,
            cursor=cursor,
            exclude_archived=True,
        )
        for ch in convs.get("channels", []):
            if ch.get("name") == value or ch.get("name_normalized") == value:
                return ch["id"]
        cursor = convs.get("response_metadata", {}).get("next_cursor") or None
        if not cursor:
            break

    # Fall back: resolve username -> user id, open IM
    # This can be imprecise if multiple users share display_name; we try exact matches
    cursor = None
    matches: list[str] = []
    while True:
        resp = client.users_list(limit=200, cursor=cursor)
        for u in resp.get("members", []):
            name_candidates = {
                u.get("name"),
                u.get("profile", {}).get("display_name"),
                u.get("profile", {}).get("display_name_normalized"),
                u.get("profile", {}).get("real_name"),
                u.get("profile", {}).get("real_name_normalized"),
            }
            if value in {c for c in name_candidates if c}:
                matches.append(u["id"])
        cursor = resp.get("response_metadata", {}).get("next_cursor") or None
        if not cursor:
            break

    if not matches:
        raise RuntimeError(f"Could not resolve destination '{destination}' to a channel or user.")
    if len(matches) > 1:
        raise RuntimeError(
            f"Username '{destination}' matched multiple users {matches}. Use email or user id."
        )
    user_id = matches[0]
    resp = client.conversations_open(users=[user_id])
    return resp["channel"]["id"]


def schedule_jobs(client: WebClient, jobs: Iterable[MessageJob]) -> list[str]:
    scheduled_ids: list[str] = []
    for job in jobs:
        assert job.channel_id is not None and job.post_at_epoch is not None
        resp = client.chat_scheduleMessage(
            channel=job.channel_id,
            text=job.content,
            post_at=job.post_at_epoch,
        )
        print(resp)
        scheduled_ids.append(resp["scheduled_message_id"])
    return scheduled_ids


@app.command()
def schedule_csv(
    csv_path: Path = typer.Argument(..., exists=True, readable=True),
    dry_run: bool = typer.Option(False, help="Parse and resolve, but do not call Slack."),
):
    """Schedule Slack messages from a CSV.

    CSV columns (required, exact headers):
      - content: message text
      - date:    'YYYY-MM-DD HH:MM' (interpreted in your local timezone)
      - destination: one of @username, email@example.com, U..., D..., C..., #channel_name

    CSV example:
      content,date,destination
      Hello from CSV,2025-10-05 09:30,@alice
      Standup,2025-10-05 10:00,#engineering
      1:1 ping,2025-10-06 14:15,bob@example.com

    Examples:
      uv run python -m meta.slack_reminders schedule-csv /abs/path/messages.csv --dry-run
      uv run python -m meta.slack_reminders schedule-csv /abs/path/messages.csv

    Environment:
      - SLACK_USER_TOKEN: User OAuth token (xoxp-...) for posting as you

    Required user scopes (minimal):
      - chat:write (schedule/post messages)
      - users:read, users:read.email (resolve users by email/name)
      - channels:read, groups:read (list/find channels by name)
      - im:write (open DMs)
    """
    token = load_user_token()
    client = WebClient(token=token)

    jobs = parse_csv(csv_path)

    # Resolve channels and compute post_at
    for job in jobs:
        try:
            job.channel_id = find_or_open_channel_id(client, job.destination)
            job.post_at_epoch = to_epoch_seconds(job.date_text)
        except SlackApiError as e:
            raise RuntimeError(
                f"Slack API error while resolving '{job.destination}': {e.response.get('error')}"
            )

    # Validate all in future (Slack requires future times)
    now = int(datetime.now().timestamp())
    in_past = [j for j in jobs if j.post_at_epoch is not None and j.post_at_epoch <= now]
    if in_past:
        examples = ", ".join([f"{j.destination} at {j.date_text}" for j in in_past[:3]])
        raise RuntimeError(f"Some messages are not in the future: {examples}")

    # Dry-run output
    for job in jobs:
        typer.echo(
            f"Ready: channel={job.channel_id} at {job.post_at_epoch} ({job.date_text}) text='{job.content[:50]}'"
        )

    if dry_run:
        typer.echo("Dry-run enabled: no messages scheduled.")
        return

    try:
        ids = schedule_jobs(client, jobs)
    except SlackApiError as e:
        raise RuntimeError(f"Slack API error while scheduling: {e.response.get('error')}")

    typer.echo(f"Scheduled {len(ids)} messages.")


def list_joined_channels(client: WebClient) -> list[dict]:
    channels: list[dict] = []
    cursor: Optional[str] = None
    while True:
        resp = client.conversations_list(
            types="public_channel,private_channel",
            exclude_archived=True,
            limit=1000,
            cursor=cursor,
        )
        channels.extend(resp.get("channels", []))
        cursor = resp.get("response_metadata", {}).get("next_cursor") or None
        if not cursor:
            break
    return channels


def fetch_channel_members(client: WebClient, channel_id: str) -> list[str]:
    member_ids: list[str] = []
    cursor: Optional[str] = None
    while True:
        resp = client.conversations_members(channel=channel_id, limit=1000, cursor=cursor)
        member_ids.extend(resp.get("members", []))
        cursor = resp.get("response_metadata", {}).get("next_cursor") or None
        if not cursor:
            break
    return member_ids


def users_by_id(client: WebClient, user_ids: list[str]) -> dict[str, dict]:
    # Slack has no bulk lookup by id; we can page users_list and filter
    users: dict[str, dict] = {}
    cursor: Optional[str] = None
    wanted = set(user_ids)
    while True and wanted:
        resp = client.users_list(limit=200, cursor=cursor)
        for u in resp.get("members", []):
            uid = u.get("id")
            if uid in wanted:
                users[uid] = u
        wanted -= set(users.keys())
        cursor = resp.get("response_metadata", {}).get("next_cursor") or None
        if not cursor:
            break
    return users


@app.command()
def export_channel_members(
    output_csv: Optional[Path] = typer.Option(
        None, help="If provided, write CSV (id,name,display_name,real_name,email)"
    )
):
    """Interactively list channels, select one, and output members with emails.

    Examples:
      uv run python -m meta.slack_reminders export-channel-members
      uv run python -m meta.slack_reminders export-channel-members --output-csv /abs/path/members.csv

    Required user scopes:
      - channels:read, groups:read (list channels)
      - users:read, users:read.email (map users to names/emails)
    """
    token = load_user_token()
    client = WebClient(token=token)

    chans = list_joined_channels(client)
    if not chans:
        raise RuntimeError("No channels available to list.")

    # Fuzzy search selection
    choices = [
        {"name": f"#{(ch.get('name') or '')} ({ch.get('id')})", "value": ch.get("id")}
        for ch in chans
    ]
    selected_channel_id = inquirer.fuzzy(
        message="Select a channel",
        choices=choices,
    ).execute()
    channel = next((c for c in chans if c.get("id") == selected_channel_id), None)
    if channel is None:
        raise RuntimeError("Selected channel not found after selection.")

    member_ids = fetch_channel_members(client, channel["id"])
    user_map = users_by_id(client, member_ids)

    rows = []
    for uid in member_ids:
        u = user_map.get(uid, {})
        profile = u.get("profile", {})
        rows.append(
            {
                "id": uid,
                "name": u.get("name"),
                "display_name": profile.get("display_name"),
                "real_name": profile.get("real_name"),
                "email": profile.get("email"),
            }
        )

    if output_csv:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "name", "display_name", "real_name", "email"]
            )
            writer.writeheader()
            writer.writerows(rows)
        typer.echo(f"Wrote {len(rows)} members to {output_csv}")
    else:
        for r in rows:
            typer.echo(
                f"{r['id']}, {r['name']}, {r['display_name']}, {r['real_name']}, {r['email']}"
            )


def pick_channel_with_fuzzy(client: WebClient) -> dict:
    chans = list_joined_channels(client)
    if not chans:
        raise RuntimeError("No channels available to list.")
    choices = [
        {"name": f"#{(ch.get('name') or '')} ({ch.get('id')})", "value": ch.get("id")}
        for ch in chans
    ]
    selected_channel_id = inquirer.fuzzy(
        message="Select a channel",
        choices=choices,
    ).execute()
    channel = next((c for c in chans if c.get("id") == selected_channel_id), None)
    if channel is None:
        raise RuntimeError("Selected channel not found after selection.")
    return channel


def list_scheduled_for_channel(client: WebClient, channel_id: str) -> list[dict]:
    scheduled: list[dict] = []
    cursor: Optional[str] = None
    while True:
        resp = client.chat_scheduledMessages_list(channel=channel_id, limit=100, cursor=cursor)
        scheduled.extend(resp.get("scheduled_messages", []))
        cursor = resp.get("response_metadata", {}).get("next_cursor") or None
        if not cursor:
            break

    return scheduled


def list_all_scheduled(client: WebClient) -> list[dict]:
    scheduled: list[dict] = []
    cursor: Optional[str] = None
    while True:
        # Without channel, Slack returns scheduled messages across channels for the authed user
        resp = client.chat_scheduledMessages_list(limit=100, cursor=cursor)
        scheduled.extend(resp.get("scheduled_messages", []))
        cursor = resp.get("response_metadata", {}).get("next_cursor") or None
        if not cursor:
            break
    return scheduled


def _resolve_channel_labels(client: WebClient, channel_ids: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    try:
        self_user = client.auth_test().get("user_id")
    except SlackApiError:
        self_user = None

    for ch_id in channel_ids:
        if not ch_id:
            continue
        try:
            info = client.conversations_info(channel=ch_id)["channel"]
            if info.get("is_im"):
                # Direct message: prefer the 'user' field on the IM channel
                other = info.get("user")
                if not other:
                    # Fallback: attempt to read members if permitted
                    try:
                        members = client.conversations_members(channel=ch_id, limit=2).get(
                            "members", []
                        )
                        other = next((m for m in members if m != self_user), None)
                    except SlackApiError:
                        other = None
                if other:
                    u = users_by_id(client, [other]).get(other, {})
                    p = u.get("profile", {})
                    name = p.get("display_name") or p.get("real_name") or u.get("name") or other
                    labels[ch_id] = f"@{name}"
                else:
                    labels[ch_id] = ch_id
            elif info.get("is_mpim"):
                # Group DM: show up to 2 names
                try:
                    members = client.conversations_members(channel=ch_id, limit=10).get(
                        "members", []
                    )
                    others = [m for m in members if m != self_user][:2]
                    users_map = users_by_id(client, others) if others else {}
                    names: list[str] = []
                    for uid in others:
                        u = users_map.get(uid, {})
                        p = u.get("profile", {})
                        n = p.get("display_name") or p.get("real_name") or u.get("name") or uid
                        names.append(f"@{n}")
                    labels[ch_id] = ",".join(names) or (info.get("name") or ch_id)
                except SlackApiError:
                    labels[ch_id] = info.get("name") or ch_id
            else:
                # Public/private channel
                labels[ch_id] = f"#{info.get('name') or ch_id}"
        except SlackApiError:
            labels[ch_id] = ch_id
    return labels


@app.command()
def delete_scheduled():
    """List all scheduled messages across channels, multi-select with fuzzy, confirm, and delete."""
    token = load_user_token()
    client = WebClient(token=token)

    msgs = list_all_scheduled(client)
    # Build channel labels for display
    unique_channel_ids = sorted({m.get("channel_id") for m in msgs if m.get("channel_id")})
    channel_id_to_label = _resolve_channel_labels(client, unique_channel_ids)

    if not msgs:
        typer.echo("No scheduled messages found for this channel.")
        return

    def fmt(m: dict) -> str:
        txt = (m.get("text") or "").replace("\n", " ")
        ts = m.get("post_at")
        when = (
            datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
            if isinstance(ts, int)
            else str(ts)
        )
        ch_id = m.get("channel_id")
        ch_label = channel_id_to_label.get(ch_id, ch_id)
        return f"[{ch_label}] {txt[:60]}...  (post_at={when})  id={m.get('id')}"

    choices = [{"name": fmt(m), "value": m["id"]} for m in msgs]
    selected_ids: list[str] = inquirer.fuzzy(
        message="Select scheduled messages to delete",
        choices=choices,
        multiselect=True,
        instruction="Use typing to filter, <tab> to toggle",
    ).execute()
    if not selected_ids:
        typer.echo("Nothing selected. Aborting.")
        return

    confirm = inquirer.confirm(
        message=f"Delete {len(selected_ids)} scheduled messages?",
        default=False,
    ).execute()
    if not confirm:
        typer.echo("Cancelled.")
        return

    errors: list[str] = []
    # Build id -> message lookup to get per-message channel_id
    id_to_msg = {m.get("id"): m for m in msgs}
    for sid in selected_ids:
        try:
            msg = id_to_msg.get(sid) or {}
            ch_id = msg.get("channel_id")
            if not ch_id:
                errors.append(f"{sid}: missing channel_id in scheduled message")
                continue
            client.chat_deleteScheduledMessage(channel=ch_id, scheduled_message_id=sid)
            typer.echo(f"Deleted {sid} (channel {ch_id})")
        except SlackApiError as e:
            errors.append(f"{sid}: {e.response.get('error')}")

    if errors:
        typer.echo("Some deletions failed:\n" + "\n".join(errors))


def relative_time(post_at_epoch: int) -> str:
    """Format time relative to now."""
    now_epoch = int(datetime.now().timestamp())
    diff_seconds = post_at_epoch - now_epoch

    if diff_seconds < 0:
        return "in the past"
    if diff_seconds < 3600:
        minutes = diff_seconds // 60
        if minutes < 1:
            return "< 1 minute"
        return f"in {minutes} minute{'s' if minutes != 1 else ''}"
    if diff_seconds < 86400:
        hours = diff_seconds // 3600
        return f"in {hours} hour{'s' if hours != 1 else ''}"

    days = diff_seconds // 86400
    remaining_hours = (diff_seconds % 86400) // 3600
    if remaining_hours == 0:
        return f"in {days} day{'s' if days != 1 else ''}"
    return f"in {days} day{'s' if days != 1 else ''} {remaining_hours} hour{'s' if remaining_hours != 1 else ''}"


class OneOnOneRemindersModel(BaseModel):
    """Pydantic model for 1:1 reminders YAML validation."""

    date: str
    template: str
    pairs: dict[str, str]
    user_id: dict[str, str]

    class Config:
        extra = "allow"  # Allow extra fields but validate required ones


def parse_1_1_yaml(yaml_path: Path) -> dict:
    """Parse a 1:1 reminders YAML file."""
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    try:
        validated = OneOnOneRemindersModel(**data)
        return validated.model_dump()
    except ValidationError as e:
        raise RuntimeError(f"Invalid YAML structure: {e}")


def schedule_1_1_reminders(
    yaml_path: Path = typer.Argument(..., exists=True, readable=True),
):
    """Schedule 1:1 reminder messages from a YAML file.

    YAML format:
      date: 'YYYY-MM-DD HH:MM'
      template: 'Message with {name1} and {name2} variables'
      pairs:
        name1: name2
        ...
      user_id:
        name1: Uxxxxxxxx
        name2: Uxxxxxxxx
        ...

    For each pair, a message is sent to name1 (the key) with name1 and name2 templated.

    Example:
      uv run python -m meta.slack_reminders schedule-1-1-reminders path/reminders.yaml
    """
    token = load_user_token()
    client = WebClient(token=token)

    data = parse_1_1_yaml(yaml_path)
    date_str = data.get("date")
    template = data.get("template", "")
    pairs: dict[str, str] = data.get("pairs", {})
    user_ids: dict[str, str] = data.get("user_id", {})

    if not date_str or not template or not pairs:
        raise RuntimeError("YAML must have 'date', 'template', and 'pairs' keys.")

    post_at_epoch = to_epoch_seconds(date_str)
    now = int(datetime.now().timestamp())
    if post_at_epoch <= now:
        raise RuntimeError(f"Date '{date_str}' is not in the future (using local timezone).")

    # Build jobs
    jobs: list[tuple[str, str, str, str]] = []  # (name1, user_id, content, relative_time)
    for name1, name2 in pairs.items():
        user_id = user_ids.get(name1)
        if not user_id:
            raise RuntimeError(f"User '{name1}' not found in user_id map.")
        content = template.format(name1=name1, name2=name2)
        rel_time = relative_time(post_at_epoch)
        jobs.append((name1, user_id, content, rel_time))

    # Display all messages with confirmation
    typer.echo(
        f"\nScheduling {len(jobs)} messages for {date_str} ({relative_time(post_at_epoch)}):\n"
    )
    for name1, user_id, content, rel_time in jobs:
        typer.echo(f"→ @{name1} ({user_id}): {content[:70]}")
    typer.echo("")

    confirm = inquirer.confirm(
        message=f"Schedule {len(jobs)} messages?",
        default=False,
    ).execute()
    if not confirm:
        typer.echo("Cancelled.")
        return

    # Schedule all
    scheduled_ids: list[str] = []
    for name1, user_id, content, _ in jobs:
        try:
            resp = client.chat_scheduleMessage(
                channel=user_id,
                text=content,
                post_at=post_at_epoch,
            )
            scheduled_ids.append(resp["scheduled_message_id"])
        except SlackApiError as e:
            raise RuntimeError(f"Slack API error for '{name1}': {e.response.get('error')}")

    typer.echo(f"\n✓ Scheduled {len(scheduled_ids)} messages.")


app.command(help=schedule_1_1_reminders.__doc__.replace("\n", "\n\n"), no_args_is_help=True)(
    schedule_1_1_reminders
)


def main():
    app()


if __name__ == "__main__":
    main()
