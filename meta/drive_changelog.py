#!/usr/bin/env python3
"""
Drive Changelog Monitor

Monitors a Google Drive folder for changes and sends summaries to Slack.
Based on the plan in drive_changelog_plan.md
"""

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any
from difflib import unified_diff
from dataclasses import dataclass

import requests
import typer
from pydantic import BaseModel, Field
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import litellm
from dotenv import load_dotenv

load_dotenv()

# Google Drive API scopes
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Configuration constants
MAX_DIFF_SIZE = 100_000  # Maximum diff size in characters before truncating
MAX_BATCH_SIZE = 100  # Maximum number of LLM requests to batch

# Typer app
app = typer.Typer(
    name="drive-changelog",
    help="Monitor Google Drive folder changes and send summaries to Slack",
    no_args_is_help=True,
)


class MonitorState(BaseModel):
    """Pydantic model for tracking monitor state"""

    last_page_token: Optional[str] = None
    file_revisions: Dict[str, "FileState"] = Field(default_factory=dict)

    def save_to_file(self, path: Path) -> None:
        """Save state to JSON file"""
        path.parent.mkdir(exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_from_file(cls, path: Path) -> "MonitorState":
        """Load state from JSON file"""
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = json.load(f)
        return cls.model_validate(data)


@dataclass
class FileChange:
    """Data class representing a file change"""

    file_id: str
    file_name: str
    change_type: str  # 'added', 'modified', 'deleted'
    summary: Optional[str] = None
    version_history_url: Optional[str] = None
    error: Optional[str] = None
    authors: Optional[List[str]] = None

    @property
    def doc_url(self) -> str:
        return f"https://docs.google.com/document/d/{self.file_id}"


@dataclass
class BatchLLMRequest:
    """Data class for batching LLM requests"""

    file_id: str
    file_name: str
    diff_text: str


class FileState(BaseModel):
    """Pydantic model for tracking a single file's state"""

    revision_id: str
    text_content: str
    file_name: str


class DriveChangelogMonitor:
    def __init__(
        self,
        credentials_path: Path,
        slack_webhook_url: str,
        folder_id: str | None = None,
        model: str = "groq/llama-3.3-70b-versatile",
        state_file: Path = Path("meta/state.json"),
    ):
        self.credentials_path = credentials_path
        self.slack_webhook_url = slack_webhook_url
        self.folder_id = folder_id
        self.model = model
        self.state_file = state_file
        self.service = None
        self.state: MonitorState | None = None  # Will be loaded in run()
        self.target_folder_ids: set[str] | None = None  # Cache for folder hierarchy checks

    def authenticate(self) -> None:
        """Authenticate with Google Drive API using service account"""
        if not self.credentials_path.exists():
            raise FileNotFoundError(f"Google credentials not found at {self.credentials_path}")

        credentials = Credentials.from_service_account_file(
            str(self.credentials_path), scopes=SCOPES
        )
        self.service = build("drive", "v3", credentials=credentials)
        print("✓ Authenticated with Google Drive API")

    def load_state(self) -> None:
        """Load state from JSON file"""
        self.state = MonitorState.load_from_file(self.state_file)
        print(f"✓ Loaded state with {len(self.state.file_revisions)} tracked files")

    def save_state(self) -> None:
        """Save state to JSON file"""
        self.state.save_to_file(self.state_file)
        print(f"✓ Saved state with {len(self.state.file_revisions)} tracked files")

    def get_file_text_content(self, file_id: str) -> str:
        """Get text content of a Google Doc"""
        # Export current version as plain text
        export_result = self.service.files().export(fileId=file_id, mimeType="text/plain").execute()

        return export_result.decode("utf-8")

    def _get_all_target_folder_ids(self) -> set[str]:
        """
        Recursively find all sub-folder IDs starting from the root folder_id.
        This is called once at the beginning of a run for efficiency.
        """
        if not self.folder_id:
            return None  # `None` means monitor all folders

        print(f"Fetching all sub-folders of root folder ID: {self.folder_id}...")
        all_folders = {self.folder_id}
        folders_to_process = [self.folder_id]

        while folders_to_process:
            # Process all folders at the current level at once
            current_batch = folders_to_process.copy()
            folders_to_process.clear()

            # Build query to find subfolders of all current folders in one API call
            parent_conditions = " or ".join(
                [f"'{folder_id}' in parents" for folder_id in current_batch]
            )
            q = f"({parent_conditions}) and mimeType = 'application/vnd.google-apps.folder'"

            page_token = None
            while True:
                response = (
                    self.service.files()
                    .list(
                        q=q,
                        spaces="drive",
                        fields="nextPageToken, files(id, name)",
                        pageToken=page_token,
                    )
                    .execute()
                )

                for folder in response.get("files", []):
                    folder_id = folder.get("id")
                    if folder_id not in all_folders:
                        all_folders.add(folder_id)
                        folders_to_process.append(folder_id)

                page_token = response.get("nextPageToken", None)
                if not page_token:
                    break

        print(f"✓ Found {len(all_folders)} total folders to monitor.")
        return all_folders

    def _perform_initial_scan(self):
        """
        On the very first run, scans all existing files to create a baseline state.
        This prevents the monitor from ignoring files that existed before it started.
        """
        print("ℹ Performing initial scan to build baseline state (this may take a while)...")

        if self.target_folder_ids is None:
            self.target_folder_ids = self._get_all_target_folder_ids()

        # Build query to find all Google Docs in the target folders.
        # If no folder is specified, target_folder_ids will be None, and we scan all.
        parent_conditions = ""
        if self.target_folder_ids:
            parent_conditions = " or ".join(
                [f"'{folder_id}' in parents" for folder_id in self.target_folder_ids]
            )
            parent_conditions = f"({parent_conditions}) and "
        q = f"{parent_conditions}mimeType = 'application/vnd.google-apps.document' and trashed = false"

        page_token = None
        files_processed = 0
        while True:
            response = (
                self.service.files()
                .list(
                    q=q,
                    spaces="drive",
                    fields="nextPageToken, files(id, name)",
                    corpora="allDrives",
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                    pageToken=page_token,
                )
                .execute()
            )

            files_on_page = response.get("files", [])
            if not files_on_page:
                print("No files found on this page to process.")
                break  # Exit if no files are returned at all

            batch_results = {}

            def batch_callback(request_id, response, exception):
                file_id, req_type = request_id.split("|", 1)
                if exception:
                    print(
                        f"    - Warning: API error for file '{batch_results.get(file_id, {}).get('name', file_id)}' ({req_type}): {exception}"
                    )
                    batch_results[file_id]["error"] = True
                    return

                if req_type == "content":
                    batch_results[file_id]["text_content"] = response.decode("utf-8")
                elif req_type == "revisions":
                    if response and response.get("revisions"):
                        batch_results[file_id]["revision_id"] = response["revisions"][-1]["id"]
                    else:
                        print(
                            f"    - Warning: No revisions found for file '{batch_results[file_id]['name']}', skipping."
                        )
                        batch_results[file_id]["error"] = True

            batch = self.service.new_batch_http_request(callback=batch_callback)

            for file_data in files_on_page:
                file_id = file_data["id"]
                file_name = file_data.get("name", "Untitled")
                batch_results[file_id] = {"name": file_name}
                batch.add(
                    self.service.files().export(fileId=file_id, mimeType="text/plain"),
                    request_id=f"{file_id}|content",
                )
                batch.add(
                    self.service.revisions().list(fileId=file_id, fields="revisions(id)"),
                    request_id=f"{file_id}|revisions",
                )

            print(f"Executing batch request for {len(files_on_page)} files...")
            batch.execute()

            for file_id, result in batch_results.items():
                file_name = result["name"]
                print(f"  - Processing state for: {file_name}")

                if (
                    result.get("error")
                    or "text_content" not in result
                    or "revision_id" not in result
                ):
                    # A warning was already printed in the batch callback
                    continue

                self.state.file_revisions[file_id] = FileState(
                    revision_id=result["revision_id"],
                    text_content=result["text_content"],
                    file_name=file_name,
                )
                files_processed += 1

            page_token = response.get("nextPageToken", None)
            if not page_token:
                break

        print(f"✓ Initial scan found and processed {files_processed} files.")

        # Finally, get the start page token for future runs.
        start_token_response = self.service.changes().getStartPageToken().execute()
        self.state.last_page_token = start_token_response["startPageToken"]

    def is_file_in_target_folder(self, file_id: str, parents: List[str]) -> bool:
        """Check if file is in target folder or any of its subfolders."""
        if self.target_folder_ids is None:
            return True  # Monitor all files if no folder specified or if lookup failed

        if not parents:
            return False

        # Fast check against the pre-fetched set of folder IDs
        return any(parent_id in self.target_folder_ids for parent_id in parents)

    def _get_revision_authors(
        self, file_id: str, last_known_revision_id: Optional[str]
    ) -> Optional[List[str]]:
        """Get the display names of all unique authors of new revisions."""
        all_revisions = (
            self.service.revisions()
            .list(fileId=file_id, fields="revisions(id,lastModifyingUser/displayName)")
            .execute()
            .get("revisions", [])
        )

        if not all_revisions:
            return None

        if last_known_revision_id:
            # Find new revisions since the last known one
            revision_ids = [rev["id"] for rev in all_revisions]
            if last_known_revision_id in revision_ids:
                last_known_index = revision_ids.index(last_known_revision_id)
                new_revisions = all_revisions[last_known_index + 1 :]
            else:
                # The revision history is too long and the old one is gone.
                # We cannot definitively determine the new authors.
                print(
                    f"Warning: Could not find last known revision for file {file_id} to determine authors."
                )
                return None
        else:
            # This is a new file, so all revisions are new.
            # In practice, there should only be one for a brand new file.
            new_revisions = all_revisions

        authors = {
            rev["lastModifyingUser"]["displayName"]
            for rev in new_revisions
            if rev.get("lastModifyingUser")
        }
        return sorted(list(authors)) if authors else None

    def generate_change_summaries(self, requests: List[BatchLLMRequest]) -> Dict[str, str]:
        """Generate change summaries for multiple files in batch using litellm."""
        if not requests:
            return {}

        # Prepare messages for batch completion
        messages = []
        for req in requests:
            # Check diff size before sending to LLM
            if len(req.diff_text) > MAX_DIFF_SIZE:
                # Add oversized diffs to the results directly, don't send to LLM
                continue

            prompt = f"""Summarize the changes to "{req.file_name}" in a single, concise sentence like a git commit message.

Requirements:
- Maximum 80 characters
- Start with a verb (e.g., "Add", "Update", "Remove", "Fix")
- Focus on the most significant changes
- Be specific but brief
- Aim for 4-12 words

Diff:
{req.diff_text}

Summary:"""
            messages.append([{"role": "user", "content": prompt}])

        if not messages:
            # Handle case where all diffs were oversized
            summaries = {}
            for req in requests:
                summaries[req.file_id] = (
                    f"Changes too large to summarize ({len(req.diff_text):,} characters)"
                )
            return summaries

        # Perform batch completion
        print(f"Sending {len(messages)} requests to LLM for batch summarization...")
        responses = litellm.batch_completion(
            model=self.model,
            messages=messages,
            max_tokens=200,
            temperature=0.3,
        )

        # Process responses
        summaries = {}
        # We need to align requests and responses. Assuming they maintain order.
        # Filter out the oversized requests from the original list to align with messages sent.
        requests_sent_to_llm = [req for req in requests if len(req.diff_text) <= MAX_DIFF_SIZE]

        for i, response in enumerate(responses):
            file_id = requests_sent_to_llm[i].file_id

            # Check if the response is an exception object
            if isinstance(response, Exception):
                print(f"Warning: LLM API error for file '{requests_sent_to_llm[i].file_name}':")
                traceback.print_exception(type(response), response, response.__traceback__)
                summaries[file_id] = "Summary generation failed due to an API error."
            elif response.choices and response.choices[0].message.content:
                summaries[file_id] = response.choices[0].message.content.strip()
            else:
                # Handle potential errors or empty responses from the batch
                summaries[file_id] = "Summary generation failed."

        # Add back the oversized diff messages
        for req in requests:
            if len(req.diff_text) > MAX_DIFF_SIZE:
                summaries[req.file_id] = (
                    f"Changes too large to summarize ({len(req.diff_text):,} characters)"
                )

        return summaries

    def fetch_changes(self) -> List[Dict[str, Any]]:
        """Fetch changes from Google Drive API with pagination support"""
        all_changes = []

        try:
            # Get changes since last page token
            request_params = {
                "includeRemoved": True,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
                "fields": "changes(file(id,name,mimeType,parents),fileId,removed),newStartPageToken,nextPageToken",
                "pageSize": 1000,  # Maximum page size
            }

            if self.state.last_page_token:
                request_params["pageToken"] = self.state.last_page_token
            else:
                # First run - get current page token to start monitoring from now
                start_token_response = self.service.changes().getStartPageToken().execute()
                self.state.last_page_token = start_token_response["startPageToken"]
                print("ℹ First run: starting monitoring from current state")
                return []

            # Handle pagination
            page_token = self.state.last_page_token
            while True:
                if page_token:
                    request_params["pageToken"] = page_token

                response = self.service.changes().list(**request_params).execute()

                changes = response.get("changes", [])
                all_changes.extend(changes)

                # Check for next page
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

                page_token = next_page_token
                print(f"Fetching next page of changes... (total so far: {len(all_changes)})")

            # Update page token for next run
            self.state.last_page_token = response["newStartPageToken"]

            print(f"✓ Fetched {len(all_changes)} changes from Drive API")
            return all_changes

        except HttpError as e:
            raise Exception(f"Failed to fetch changes from Drive API: {e}")

    def is_target_file(self, file_data: Dict[str, Any]) -> bool:
        """Check if file is a Google Doc in our target folder"""
        if not file_data or file_data.get("mimeType") != "application/vnd.google-apps.document":
            return False

        # Use explicit dict access for required fields
        parents = file_data["parents"]  # Will raise KeyError if missing
        file_id = file_data["id"]  # Will raise KeyError if missing

        return self.is_file_in_target_folder(file_id, parents)

    def process_changes(self, changes: List[Dict[str, Any]]) -> List[FileChange]:
        """Process changes and return structured data (no formatting)"""
        file_changes = []
        llm_requests = []

        for change in changes:
            if change["removed"]:
                file_id = change["fileId"]
                if file_id in self.state.file_revisions:
                    # Known file was deleted.
                    deleted_file_name = self.state.file_revisions[file_id].file_name
                    file_changes.append(
                        FileChange(
                            file_id=file_id, file_name=deleted_file_name, change_type="deleted"
                        )
                    )
                    del self.state.file_revisions[file_id]
                else:
                    # Untracked file was deleted.
                    file_changes.append(
                        FileChange(
                            file_id=file_id,
                            file_name="Unknown file",
                            change_type="deleted",
                            error="File was deleted, but not tracked by the monitor.",
                        )
                    )
                continue

            file_data = change.get("file")
            if not file_data:
                # Not a file change, so we can ignore it (e.g., change to a drive's metadata)
                print(
                    f"ℹ Skipping non-file change event for file ID: {change.get('fileId', 'N/A')}"
                )
                print(change)
                continue

            # Case 0: File was trashed - this is a form of deletion
            if file_data.get("trashed"):
                file_id = file_data["id"]
                if file_id in self.state.file_revisions:
                    deleted_file_name = self.state.file_revisions[file_id].file_name
                    file_changes.append(
                        FileChange(
                            file_id=file_id, file_name=deleted_file_name, change_type="deleted"
                        )
                    )
                    del self.state.file_revisions[file_id]
                else:
                    # Untracked file was trashed.
                    file_changes.append(
                        FileChange(
                            file_id=file_id,
                            file_name="Unknown file",
                            change_type="deleted",
                            error="File was trashed, but not tracked by the monitor.",
                        )
                    )
                continue

            if not self.is_target_file(file_data):
                print(
                    f"ℹ Skipping non-target file: '{file_data.get('name', 'Untitled')}' (ID: {file_data.get('id')}, MIME Type: {file_data.get('mimeType')})"
                )
                print(file_data)
                continue

            file_id = file_data["id"]
            file_name = file_data["name"]

            try:
                # Get current revision ID
                revisions_response = self.service.revisions().list(fileId=file_id).execute()
                if not revisions_response.get("revisions"):
                    print(f"Warning: No revisions found for file {file_name}, skipping.")
                    continue
                current_revision = revisions_response["revisions"][-1]["id"]

                # Case 1: New file
                if file_id not in self.state.file_revisions:
                    print(f"Processing new file: {file_name}")
                    new_text = self.get_file_text_content(file_id)
                    self.state.file_revisions[file_id] = FileState(
                        revision_id=current_revision, text_content=new_text, file_name=file_name
                    )

                    author_list = self._get_revision_authors(file_id, None)

                    file_changes.append(
                        FileChange(
                            file_id=file_id,
                            file_name=file_name,
                            change_type="added",
                            authors=author_list,
                        )
                    )

                # Case 2: Potentially modified file
                else:
                    file_state = self.state.file_revisions[file_id]
                    last_known_revision = file_state.revision_id

                    if current_revision != last_known_revision:
                        print(f"Processing changes for: {file_name}")

                        # Get authors of the new revisions
                        author_list = self._get_revision_authors(file_id, last_known_revision)

                        old_text = file_state.text_content
                        new_text = self.get_file_text_content(file_id)

                        # Generate diff
                        diff_lines = list(
                            unified_diff(
                                old_text.splitlines(keepends=True),
                                new_text.splitlines(keepends=True),
                                fromfile="previous version",
                                tofile="current version",
                            )
                        )

                        change_summary = None
                        if diff_lines:
                            diff_text = "".join(diff_lines)
                            llm_requests.append(
                                BatchLLMRequest(
                                    file_id=file_id, file_name=file_name, diff_text=diff_text
                                )
                            )
                        else:
                            change_summary = "Non-textual changes (e.g. formatting)."

                        file_changes.append(
                            FileChange(
                                file_id=file_id,
                                file_name=file_name,
                                change_type="modified",
                                summary=change_summary,  # Will be None for textual changes until LLM summary is added
                                version_history_url=f"https://docs.google.com/document/d/{file_id}/history",
                                authors=author_list,
                            )
                        )

                        # Update state with new content and revision
                        file_state.revision_id = current_revision
                        file_state.text_content = new_text
                        file_state.file_name = file_name  # Update name in case it changed
                    else:
                        file_changes.append(
                            FileChange(
                                file_id=file_id,
                                file_name=file_name,
                                change_type="modified",
                                summary="Metadata changes (title, folder, etc.)",
                                version_history_url=f"https://docs.google.com/document/d/{file_id}/history",
                            )
                        )

            except HttpError as e:
                print(f"Warning: API error processing file {file_name} (ID: {file_id}): {e}")
                file_changes.append(
                    FileChange(
                        file_id=file_id,
                        file_name=file_name,
                        change_type="modified",
                        error=f"Failed to process changes due to API error: {e}",
                        version_history_url=f"https://docs.google.com/document/d/{file_id}/history",
                    )
                )
                continue

        # Generate LLM summaries in batch
        if llm_requests:
            print(f"Generating {len(llm_requests)} change summaries...")
            summaries = self.generate_change_summaries(llm_requests)

            # Update file changes with summaries
            for file_change in file_changes:
                if file_change.change_type == "modified" and file_change.file_id in summaries:
                    file_change.summary = summaries[file_change.file_id]

        return file_changes

    def format_slack_message(self, file_changes: List[FileChange]) -> Dict[str, Any]:
        """Format file changes into Slack message payload"""
        if not file_changes:
            return {}

        # Group changes by type
        changes_by_type = {"added": [], "modified": [], "deleted": []}

        for change in file_changes:
            changes_by_type[change.change_type].append(change)

        # Build message blocks
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": "Workshop materials changes"}}
        ]

        for change_type, items in changes_by_type.items():
            if not items:
                continue

            change_lines = []
            for item in items:
                doc_link = f"<{item.doc_url}|{item.file_name}>"
                author_text = f" by {', '.join(item.authors)}" if item.authors else ""

                if change_type == "added":
                    change_lines.append(f"➕ {doc_link}{author_text}")
                elif change_type == "deleted":
                    if item.error:
                        change_lines.append(f"🗑️ *{doc_link}* ⚠️ _{item.error}_")
                    else:
                        change_lines.append(f"🗑️ {doc_link}")
                else:  # modified
                    history_link = f"<{item.version_history_url}|diff>"

                    summary_text = ""
                    if item.error:
                        summary_text = f" ⚠️ _{item.error}_"
                    elif item.summary:
                        summary_text = f" _{item.summary}_"
                    else:
                        summary_text = " _Unknown changes (style, formatting, comments, etc.)_"

                    change_lines.append(
                        f"✏️ *{doc_link}* ({history_link}){author_text}: {summary_text}"
                    )

            if change_lines:
                blocks.append(
                    {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(change_lines)}}
                )

        return {"blocks": blocks}

    def send_slack_notification(self, payload: Dict[str, Any]) -> None:
        """Send formatted notification to Slack"""
        if not payload:
            print("ℹ No changes to report")
            return

        try:
            response = requests.post(self.slack_webhook_url, json=payload)
            response.raise_for_status()
            print("✓ Sent notification to Slack")
        except requests.RequestException as e:
            raise Exception(f"Failed to send Slack notification: {e}")

    def send_error_notification(self, error_message: str) -> None:
        """Send error notification to Slack"""
        if not self.slack_webhook_url:
            return

        payload = {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "🚨 Drive Changelog Error"},
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"The drive changelog monitor encountered an error:\n```{error_message}```",
                    },
                },
            ]
        }

        requests.post(self.slack_webhook_url, json=payload)

    def run(self) -> None:
        """Main execution function"""
        try:
            print("🚀 Starting Drive Changelog Monitor")

            # Initialize
            self.authenticate()
            self.load_state()

            # If this is the first run, perform an initial scan and exit.
            # Monitoring for changes will begin on the next run.
            if not self.state.last_page_token:
                self._perform_initial_scan()
                self.save_state()
                print(
                    "✅ Initial scan complete. State populated. Monitoring will begin on the next run."
                )
                return

            # Pre-fetch all folder IDs for efficient checking
            self.target_folder_ids = self._get_all_target_folder_ids()

            # Fetch and process changes
            changes = self.fetch_changes()
            if not changes:
                print("ℹ No changes to process")
                self.save_state()
                return

            file_changes = self.process_changes(changes)

            if file_changes:
                # Format and send notification
                slack_payload = self.format_slack_message(file_changes)
                self.send_slack_notification(slack_payload)
            else:
                print("ℹ No relevant changes found")

            # Save updated state
            self.save_state()

            print("✅ Drive changelog monitoring completed successfully")

        except Exception as e:
            error_msg = (
                f"Drive changelog monitor failed: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            )
            print(f"❌ {error_msg}")
            self.send_error_notification(error_msg)
            sys.exit(1)


@app.command()
def main(
    credentials_path: Path = typer.Option(
        default=lambda: Path(os.getenv("GOOGLE_CREDENTIALS_PATH", "secret/credentials.json")),
        help="Path to Google service account credentials JSON file",
    ),
    slack_webhook_url: str = typer.Option(
        default=lambda: os.getenv("SLACK_WEBHOOK_URL", ""), help="Slack incoming webhook URL"
    ),
    folder_id: Optional[str] = typer.Option(
        default=lambda: os.getenv("DRIVE_FOLDER_ID"),
        help="Google Drive folder ID to monitor (optional - monitors all accessible docs if not specified)",
    ),
    model: str = typer.Option(
        default=lambda: os.getenv("LITELLM_MODEL", "groq/llama-3.3-70b-versatile"),
        help="LLM model to use for change summaries",
    ),
    state_file: Path = typer.Option(
        default=Path("meta/state.json"), help="Path to state file for tracking changes"
    ),
):
    """
    Monitor Google Drive folder for changes and send summaries to Slack.

    This command will:
    1. Check for new, modified, or deleted Google Docs
    2. Generate LLM-powered summaries of text changes
    3. Send formatted notifications to Slack
    4. Maintain state between runs for efficient monitoring
    """

    # Validate required parameters
    if not slack_webhook_url:
        typer.echo("❌ Error: Slack webhook URL is required", err=True)
        typer.echo(
            "Set SLACK_WEBHOOK_URL environment variable or use --slack-webhook-url option", err=True
        )
        raise typer.Exit(1)

    if not credentials_path.exists():
        typer.echo(f"❌ Error: Google credentials file not found at {credentials_path}", err=True)
        typer.echo(
            "Set GOOGLE_CREDENTIALS_PATH environment variable or use --credentials-path option",
            err=True,
        )
        raise typer.Exit(1)

    # Create and run monitor
    monitor_instance = DriveChangelogMonitor(
        credentials_path=credentials_path,
        slack_webhook_url=slack_webhook_url,
        folder_id=folder_id,
        model=model,
        state_file=state_file,
    )

    monitor_instance.run()


if __name__ == "__main__":
    app()
