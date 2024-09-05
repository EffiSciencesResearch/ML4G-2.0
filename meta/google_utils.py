import re
from typing import Literal
import yaml
import typer
from google.oauth2 import service_account
from googleapiclient.discovery import build
from pathlib import Path


def extract_id_from_url(url: str) -> str:
    """Extract the ID from a Google Slides/Docs/Sheets/... or Drive Folder URL."""
    # This should work both for urls from docs that are open and from urls of drive files
    # Urls are of the form: https://.../d/<file_id>/edit
    # Or https://drive.google.com/drive/u/0/folders/19W8SWqvprQyE_XcH1rDpGmslKCz-NEQg
    match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
    if match:
        return match.group(1)

    match = re.search(r'/folders/([a-zA-Z0-9-_]+)', url)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract ID from URL: {url}")


class SimpleGoogleAPI:
    def __init__(self, service_account_file: str):
        self.service_account_file = service_account_file

        scopes = ['https://www.googleapis.com/auth/documents', 'https://www.googleapis.com/auth/drive']
        self.credentials = service_account.Credentials.from_service_account_file(service_account_file, scopes=scopes)

        self.drive_service = build('drive', 'v3', credentials=self.credentials)
        self.docs_service = build('docs', 'v1', credentials=self.credentials)


    def get_file_name(self, file_id: str) -> str:
        """Retrieve the original presentation name using its ID."""
        file_metadata = self.drive_service.files().get(fileId=file_id, fields='name').execute()
        return file_metadata['name']


    def copy_file(self, file_id: str, folder_id: str, new_name: str | None = None) -> str:
        """Copy a presentation to a specified folder with a new name prefix."""
        if new_name is None:
            new_name = self.get_file_name(file_id)

        copied_file = self.drive_service.files().copy(
            fileId=file_id,
            body={
                'name': new_name,
                'parents': [folder_id]
            }
        ).execute()

        return copied_file['id']

    def replace_in_document(self, document_id: str, initial_text: str, new_text: str) -> dict:
        requests = [
            {
                'replaceAllText': {
                    'containsText': {
                        'text': initial_text,
                        'matchCase': True,
                    },
                    'replaceText': new_text,
                }
            }
        ]
        result = self.docs_service.documents().batchUpdate(documentId=document_id, body={'requests': requests}).execute()
        return result

    def share_document(self, file_id: str, email: str):
        """Add a user as a writer to a document."""
        user_permission = {
            'type': 'user',
            'role': 'writer',
            'emailAddress': email
        }
        self.drive_service.permissions().create(
            fileId=file_id,
            body=user_permission,
            fields='id'
        ).execute()




if __name__ == "__main__":
    api = SimpleGoogleAPI('./meta/service_account_token.json')
    print(api.get_file_name("15lHgF6-D1qemtmPaxyTN_Dfm-L91jGRr1njO7fYhRQc"))

    "https://docs.google.com/presentation/d/15lHgF6-D1qemtmPaxyTN_Dfm-L91jGRr1njO7fYhRQc/edit#slide=id.p"
