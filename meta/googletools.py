import csv
from pathlib import Path
from typing import Annotated
import typer
from googleapiclient.discovery import build
from google.oauth2 import service_account

app = typer.Typer()

# Function to replace [NAME] in the document
def replace_name_in_document(docs_service, document_id, name):
    requests = [
        {
            'replaceAllText': {
                'containsText': {
                    'text': '[NAME]',
                    'matchCase': True,
                },
                'replaceText': name,
            }
        }
    ]
    result = docs_service.documents().batchUpdate(documentId=document_id, body={'requests': requests}).execute()
    return result

# Function to copy the document
def copy_document(drive_service, file_id, name, folder_id):
    """
    Copies a Google Docs template, replacing [NAME] in the filename with the actual name.

    Args:
        drive_service: The Google Drive service instance.
        file_id: The ID of the Google Docs template to copy.
        name: The name to replace [NAME] with in the filename.
        folder_id: The ID of the Google Drive folder where the copy will be saved.

    Returns:
        The ID of the copied document.
    """
    # Get the original file metadata
    original_file = drive_service.files().get(fileId=file_id, fields='name').execute()
    original_name = original_file['name']

    # Replace [NAME] in the original name with the actual name
    new_name = original_name.replace('[NAME]', name)

    # Prepare the request body for copying the file
    body = {
        'name': new_name,
        'parents': [folder_id]
    }

    # Copy the file
    copied_file = drive_service.files().copy(fileId=file_id, body=body).execute()
    return copied_file['id']

# Function to share the document
def share_document(drive_service, file_id, email):
    user_permission = {
        'type': 'user',
        'role': 'writer',
        'emailAddress': email
    }
    drive_service.permissions().create(
        fileId=file_id,
        body=user_permission,
        fields='id'
    ).execute()

# Main function to process the document for each (name, email) pair
def process_document(template_id, folder_id, service_account_file, name_email_list):
    SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/documents']
    credentials = service_account.Credentials.from_service_account_file(service_account_file, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=credentials)
    docs_service = build('docs', 'v1', credentials=credentials)

    for name, email in name_email_list:
        copied_doc_id = copy_document(drive_service, template_id, name, folder_id)
        replace_name_in_document(docs_service, copied_doc_id, name)
        share_document(drive_service, copied_doc_id, email)
        print(f'Document for {name} shared with {email}')

@app.command()
def main(
    names_and_email_file: Annotated[Path, typer.Argument(help="Path to the CSV file containing names and emails without header")],
    template_id: Annotated[str, typer.Argument(help="Google Docs template ID")],
    folder_id: Annotated[str, typer.Argument(help="Google Drive folder ID where copies will be saved")],
    service_account_file: Annotated[Path, typer.Argument(help="Path to the service account JSON key file")]
):
    """
    Create personalized copies of a Google Docs template and share them via email.

    This script reads a CSV file containing names and emails, creates a copy of the
    Google Docs template for each name, replaces [NAME] in the document and filename,
    and shares the document with the corresponding email.
    """
    name_email_list = []
    with open(names_and_email_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            name_email_list.append((row[0], row[1]))

    process_document(template_id, folder_id, service_account_file, name_email_list)

if __name__ == "__main__":
    app()
