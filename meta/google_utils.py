# %%
from __future__ import annotations

import re
from google.oauth2 import service_account
from googleapiclient.discovery import build
import requests
from pprint import pprint
from textwrap import indent
from pydantic import BaseModel


def extract_id_from_url(url: str) -> str:
    """Extract the ID from a Google Slides/Docs/Sheets/... or Drive Folder URL."""
    # This should work both for urls from docs that are open and from urls of drive files
    # Urls are of the form: https://.../d/<file_id>/edit
    # Or https://drive.google.com/drive/u/0/folders/19W8SWqvprQyE_XcH1rDpGmslKCz-NEQg
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    if match:
        return match.group(1)

    match = re.search(r"/folders/([a-zA-Z0-9-_]+)", url)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract ID from URL: {url}")


class GDriveFileInfo(BaseModel):
    id: str
    name: str
    # Aliased to mimeType to match the API response
    mime_type: str
    children: None | list[GDriveFileInfo] = None

    @property
    def drive_url(self) -> str:
        return f"https://drive.google.com/drive/u/0/folders/{self.id}"

    @property
    def is_folder(self) -> bool:
        return self.mime_type == "application/vnd.google-apps.folder"

    def fetch_children(self, api: SimpleGoogleAPI):
        if not self.is_folder:
            return

        if self.children is None:
            self.children = api.get_tree_slow(self.id)
        else:
            # The tree might have been partially fetched before.
            for child in self.children:
                child.fetch_children(api)

    def tree(self) -> str:
        base = f"{self.name} ({self.drive_url})"
        if self.children is None:
            return base
        else:
            return base + "\n" + indent("\n".join(child.tree() for child in self.children), "| ")


class SimpleGoogleAPI:
    def __init__(self, service_account_file: str):
        self.service_account_file = service_account_file

        scopes = [
            "https://www.googleapis.com/auth/documents",
            "https://www.googleapis.com/auth/drive",
        ]
        self.credentials = service_account.Credentials.from_service_account_file(
            service_account_file, scopes=scopes
        )

        self.drive_service = build("drive", "v3", credentials=self.credentials)
        self.docs_service = build("docs", "v1", credentials=self.credentials)

    def get_file_name(self, file_id: str) -> str:
        """Retrieve the original presentation name using its ID."""
        file_metadata = self.drive_service.files().get(fileId=file_id, fields="name").execute()
        return file_metadata["name"]

    def copy_file(self, file_id: str, folder_id: str, new_name: str | None = None) -> str:
        """Copy a presentation to a specified folder with a new name prefix."""
        if new_name is None:
            new_name = self.get_file_name(file_id)

        copied_file = (
            self.drive_service.files()
            .copy(fileId=file_id, body={"name": new_name, "parents": [folder_id]})
            .execute()
        )

        return copied_file["id"]

    def replace_in_document(
        self,
        document_id: str,
        initial_text: str,
        new_text: str,
        match_case: bool = True,
    ) -> dict:
        requests = [
            {
                "replaceAllText": {
                    "containsText": {
                        "text": initial_text,
                        "matchCase": match_case,
                    },
                    "replaceText": new_text,
                }
            }
        ]
        result = (
            self.docs_service.documents()
            .batchUpdate(documentId=document_id, body={"requests": requests})
            .execute()
        )
        return result

    def share_document(self, file_id: str, email: str):
        """Add a user as a writer to a document."""
        user_permission = {"type": "user", "role": "writer", "emailAddress": email}
        self.drive_service.permissions().create(
            fileId=file_id, body=user_permission, fields="id"
        ).execute()

    def export_gdoc_as_markdown(self, document_id: str) -> str:
        """Return the content of a Google Doc as markdown."""

        # We use requests to the export link to get the markdown content directly
        export_link = (
            f"https://www.googleapis.com/drive/v3/files/{document_id}/export?mimeType=text/markdown"
        )

        # Use access token to authenticate
        headers = {
            "Authorization": f"Bearer {self.credentials.token}",
        }
        response = requests.get(export_link, headers=headers)
        response.raise_for_status()
        return response.text

    def get_shared_with_me(self) -> list[GDriveFileInfo]:
        """Get the list of folders shared with the service account."""

        results = (
            self.drive_service.files()
            .list(
                q="sharedWithMe",
                # Use just "files" to get all the metadata. It includes also permissions.
                fields="files(id, name, mimeType)",
                pageSize=1000,
            )
            .execute()
        )

        files = results["files"]
        if len(files) == 1000:
            raise ValueError("Too many files shared with the service account. We need to paginate.")

        for file in files:
            file["mime_type"] = file.pop("mimeType")

        return [GDriveFileInfo.model_validate(file) for file in files]

    def get_tree_slow(self, folder_id: str, name: str = None) -> list[GDriveFileInfo]:
        """Get the tree of files and folders inside a folder."""
        # Note: name is only used for loggin purposes
        if name is None:
            name = "/"

        print(f"Getting tree for folder {name} ({folder_id})")
        children = self.get_direct_children(folder_id)

        for file in children:
            if file.is_folder:
                file.children = self.get_tree_slow(file.id, file.name)

        return children

    def get_direct_children(self, folder_id: str) -> list[GDriveFileInfo]:
        results = (
            self.drive_service.files()
            .list(
                q=f"'{folder_id}' in parents",
                fields="files(id, name, mimeType)",
                pageSize=1000,
            )
            .execute()
        )

        files = results["files"]
        if len(files) == 1000:
            raise ValueError("Too many files in the folder. We need to paginate.")

        for file in files:
            file["mime_type"] = file.pop("mimeType")

        return [GDriveFileInfo.model_validate(file) for file in files]


if __name__ == "__main__":
    api = SimpleGoogleAPI("../meta/service_account_token.json")

    # doc_id = "1b_0XbG1X4oz7WW5iB_Ck_k-tQNaScvWhQvtdZsGSZHg"
    # print(api.get_file_name(doc_id))
    # print(api.export_gdoc_as_markdown(doc_id))

    r = api.get_shared_with_me()
    pprint(r)
    # %%
    for i, file in enumerate(r):
        print(i, file.name, file.drive_url)
    # %%
    workshops = r[4]
    workshops.fetch_children(api)
    # %%
    print(workshops.tree())

# %%
