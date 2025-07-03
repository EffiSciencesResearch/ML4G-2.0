#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from typing import Annotated
import yaml
import typer
from pathlib import Path

from utils.google_utils import extract_id_from_url, SimpleGoogleAPI
from InquirerPy import inquirer


app = typer.Typer(add_completion=False, no_args_is_help=True)

DEFAULT_SERVICE_ACCOUNT_FILE = Path(__file__).parent / "service_account_token.json"
CONFIG_FILE = Path(__file__).parent / "config.yaml"
API: SimpleGoogleAPI = None


@app.callback()
def callback(
    service_account_file: str = typer.Option(
        DEFAULT_SERVICE_ACCOUNT_FILE, help="Path to the service account file"
    )
):
    global API
    API = SimpleGoogleAPI(service_account_file)


def load_config():
    """Load configuration from a YAML file."""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except Exception:
        return {}


def save_config(config):
    """Save configuration to a YAML file."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)


@app.command(no_args_is_help=True)
def copy_to_camp_folder(
    # fmt: off
    url: str,
    folder_url: Annotated[str, typer.Option(help="Google Drive folder URL to copy the presentation to")] = None,
    camp_prefix: Annotated[str, typer.Option(help="Prefix to add to the copied presentation name")] = None
    # fmt: on
):
    """
    Copy a Google Slides/Docs/... to a specified Google Drive folder with a new name prefix.

    This is meant to be used during each camp to make the presentations public easily.

    Configuration File (config.yaml):
    - The tool maintains a configuration file named 'config.yaml' in the same directory as the script.
    - This file stores the last used folder URL and prefix, to reuse things during the camp, without needing to specify these parameters again.

    Service Account File:
    - A service account file is required to authenticate with the Google Drive API. It is a JSON file.
    - The service account must have access to both the source presentation (read) and the destination
        folder (write) in Google Drive.
    - Ask Diego for the service account file.
    """

    # Load configuration
    config = load_config()

    # Use the last folder URL from the config if not provided
    if not folder_url:
        folder_url = config.get("last_folder_url", None)
        if not folder_url:
            raise ValueError(
                "Folder URL must be provided either as an argument or in the config file."
            )
    if not camp_prefix:
        camp_prefix = config.get("camp_prefix", "")

    # Extract IDs from URLs
    presentation_id = extract_id_from_url(url)
    folder_id = extract_id_from_url(folder_url)

    # Copy the presentation directly into the specified folder
    name = API.get_file_name(presentation_id)
    new_name = camp_prefix + name

    if not inquirer.confirm(f"Copy with name '{new_name}'?", default=True):
        exit(1)

    new_id = API.copy_file(presentation_id, folder_id, new_name)

    # Print the URLs of the folder and the new presentation
    print(f"Folder URL: {folder_url}")
    new_presentation_url = url.replace(presentation_id, new_id)
    print(f"New Presentation URL: {new_presentation_url}")

    # Update the configuration with the last used folder URL
    config["last_folder_url"] = folder_url
    config["camp_prefix"] = camp_prefix
    save_config(config)


@app.command(no_args_is_help=True)
def duplicate_career_docs(
    # fmt: off
    names_and_email_file: Annotated[Path, typer.Argument(help="Path to the CSV file containing names and emails without header")],
    template_url: Annotated[str, typer.Argument(help="Google Docs template Url")],
    folder_url: Annotated[str, typer.Argument(help="Google Drive folder ID where copies will be saved")],
    # fmt: on
):
    """
    Create personalized copies of a Google Docs template and share them via email.

    This script reads a CSV file containing two columns with header, "name" and "email",
    creates a copy of the Google Docs template for each name,
    replaces [NAME] in the document and filename, and shares the document with
    the corresponding email.
    """

    email_to_name = {}
    with open(names_and_email_file, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            email_to_name[row["email"]] = row["name"]

    template_id = extract_id_from_url(template_url)
    folder_id = extract_id_from_url(folder_url)

    doc_name = API.get_file_name(template_id)
    for email, name in email_to_name.items():
        print(f"Processing document for {name} ({email})", end="... ", flush=True)
        new_name = doc_name.replace("[NAME]", name)
        copied_doc_id = API.copy_file(template_id, folder_id, new_name)
        API.replace_in_document(copied_doc_id, "[NAME]", name)
        API.share_document(copied_doc_id, email)
        print("✅")


@app.command(no_args_is_help=True)
def add_prefix_to_folders(
    # fmt: off
    folder_url: Annotated[str, typer.Argument(help="Google Drive folder URL containing the folders to rename")],
    prefix: Annotated[str, typer.Argument(help="Prefix to add to folder names")],
    dry_run: Annotated[bool, typer.Option(help="Show what would be done without making changes")] = True
    # fmt: on
):
    """
    Add a prefix to all folders within a specified Google Drive folder.

    The command will only add the prefix to folders that don't already have it.
    Use --no-dry-run to actually perform the changes.
    """
    folder_id = extract_id_from_url(folder_url)

    # Get all folders in the specified folder
    folders = API.list_folders(folder_id)

    if not folders:
        print("No folders found in the specified directory.")
        return

    changes_needed = []
    for folder in folders:
        current_name = folder.name
        if not current_name.startswith(prefix):
            new_name = prefix + current_name
            changes_needed.append((folder.id, current_name, new_name))

    if not changes_needed:
        print("No folders need to be renamed - all folders already have the prefix.")
        return

    print(f"\nFound {len(changes_needed)} folders to rename:")
    for _, old_name, new_name in changes_needed:
        print(f"  {old_name} -> {new_name}")

    if dry_run:
        print("\nThis was a dry run. Use --no-dry-run to apply the changes.")
        return

    if not inquirer.confirm("\nProceed with renaming?", default=True):
        print("Operation cancelled.")
        return

    print("\nRenaming folders...")
    for folder_id, old_name, new_name in changes_needed:
        print(f"Renaming '{old_name}' to '{new_name}'...", end=" ", flush=True)
        API.rename_file(folder_id, new_name)
        print("✅")

    print("\nAll folders have been renamed successfully!")


if __name__ == "__main__":
    app()
