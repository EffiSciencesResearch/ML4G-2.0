import re
import yaml
import typer
from google.oauth2 import service_account
from googleapiclient.discovery import build
from pathlib import Path

app = typer.Typer(add_completion=False, no_args_is_help=True)

DEFAULT_SERVICE_ACCOUNT_FILE = './service_account_token.json'
CONFIG_FILE = Path(__file__).parent / 'config.yaml'

def load_config():
    """Load configuration from a YAML file."""
    if Path(CONFIG_FILE).exists():
        with open(CONFIG_FILE, 'r') as file:
            return yaml.safe_load(file)
    else:
        return {}

def save_config(config):
    """Save configuration to a YAML file."""
    with open(CONFIG_FILE, 'w') as file:
        yaml.safe_dump(config, file)

def get_service(service_account_file: str):
    """Authenticate and build the Google Drive service."""
    print("Authenticating and building the Drive service...")
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=['https://www.googleapis.com/auth/drive'])
    return build('drive', 'v3', credentials=credentials)

def extract_id_from_url(url: str, type: str = 'presentation') -> str:
    """Extract the ID from a Google Slides or Drive URL."""
    pattern = r'/d/([a-zA-Z0-9-_]+)' if type == 'presentation' else r'/folders/([a-zA-Z0-9-_]+)'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Invalid {type} URL: {url}")

def get_presentation_name(service, presentation_id: str) -> str:
    """Retrieve the original presentation name using its ID."""
    file_metadata = service.files().get(fileId=presentation_id, fields='name').execute()
    return file_metadata['name']

def copy_presentation(service, presentation_id: str, folder_id: str, camp_prefix: str) -> str:
    """Copy a presentation to a specified folder with a new name prefix."""
    original_name = get_presentation_name(service, presentation_id)
    new_name = camp_prefix + original_name
    print(f"Copying the presentation with name: {new_name}...")
    if input("Press Enter to continue or Ctrl+C to cancel..."):
        print("Operation cancelled.")
        quit()

    copied_file = service.files().copy(
        fileId=presentation_id,
        body={
            'name': new_name,
            'parents': [folder_id]
        }
    ).execute()

    return copied_file['id']

@app.command()
def main(
    presentation_url: str,
    folder_url: str = typer.Option(None, help="Google Drive folder URL"),
    camp_prefix: str = typer.Option("", help="Prefix to add to the new presentation name"),
    service_account_file: str = typer.Option(DEFAULT_SERVICE_ACCOUNT_FILE, help="Path to the service account file")
):
    """
    Copy a Google Slides presentation to a specified Google Drive folder with a new name prefix.

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
        folder_url = config.get('last_folder_url', None)
        if not folder_url:
            raise ValueError("Folder URL must be provided either as an argument or in the config file.")
    if not camp_prefix:
        camp_prefix = config.get('camp_prefix', '')

    # Extract IDs from URLs
    presentation_id = extract_id_from_url(presentation_url, 'presentation')
    folder_id = extract_id_from_url(folder_url, 'folder')

    # Get the Google Drive service
    service = get_service(service_account_file)

    # Copy the presentation directly into the specified folder
    copied_presentation_id = copy_presentation(service, presentation_id, folder_id, camp_prefix)

    # Print the URLs of the folder and the new presentation
    print(f"Folder URL: {folder_url}")
    new_presentation_url = f"https://docs.google.com/presentation/d/{copied_presentation_id}/edit"
    print(f"New Presentation URL: {new_presentation_url}")

    # Update the configuration with the last used folder URL
    config['last_folder_url'] = folder_url
    config['camp_prefix'] = camp_prefix
    save_config(config)

if __name__ == '__main__':
    app()
