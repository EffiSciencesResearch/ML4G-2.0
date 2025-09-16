from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import yaml
import os
import pickle

from models import (
    AnyQuestionConfig,
    CampConfig,
    ChoiceQuestionConfig,
    ParagraphQuestionConfig,
    ScaleQuestionConfig,
    TextQuestionConfig,
)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CREDS_PATH = os.path.join(SCRIPT_DIR, "creds.json")
TOKEN_PATH = os.path.join(SCRIPT_DIR, "token.pickle")
SCOPES = [
    "https://www.googleapis.com/auth/forms.body",
    "https://www.googleapis.com/auth/drive.file",
]


def get_credentials():
    """Get credentials with service account fallback to OAuth2."""
    # Try service account first if available
    service_account_path = os.path.join(os.path.dirname(SCRIPT_DIR), "service_account_token.json")
    if os.path.exists(service_account_path):
        print("Using service account authentication...")
        return service_account.Credentials.from_service_account_file(
            service_account_path, scopes=SCOPES
        )

    # Fall back to existing OAuth2 flow
    print("Using OAuth2 authentication...")
    creds = None

    # Try to load existing token
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
                creds = flow.run_local_server(port=0)
            except FileNotFoundError:
                print("\n❌ Missing Google credentials!")
                print(f"Service account file not found: {service_account_path}")
                print(f"OAuth2 credentials not found: {CREDS_PATH}")
                print("\nTo use service account:")
                print("1. Place service_account_token.json in meta/ directory")
                print("\nTo use OAuth2:")
                print("1. Go to https://console.cloud.google.com/apis/credentials")
                print("2. Create OAuth 2.0 Client ID (Desktop app)")
                print("3. Download JSON and save as creds.json in meta/feedback_forms/")
                raise

        # Save the credentials for the next run
        with open(TOKEN_PATH, "wb") as token:
            pickle.dump(creds, token)

    return creds


def get_forms_service():
    """Authenticate and return Google Forms service."""
    creds = get_credentials()
    return build("forms", "v1", credentials=creds)


def get_drive_service():
    """Authenticate and return Google Drive service."""
    creds = get_credentials()
    return build("drive", "v3", credentials=creds)


def load_config(config_path: str) -> CampConfig:
    """Load configuration from YAML file and validate with Pydantic."""
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)
        return CampConfig.model_validate(config_data)


def move_file_to_folder(drive_service, file_id, folder_id):
    """Move a file to a specific folder in Google Drive."""
    # Retrieve the file to get its parents
    file = drive_service.files().get(fileId=file_id, fields="parents").execute()
    previous_parents = ",".join(file.get("parents"))

    # Move the file to the new folder
    drive_service.files().update(
        fileId=file_id, addParents=folder_id, removeParents=previous_parents, fields="id, parents"
    ).execute()


def create_base_form(service, title, description=None):
    """Create a new form with the given title and optional description."""
    # First create the form with just the title
    form = service.forms().create(body={"info": {"title": title, "documentTitle": title}}).execute()

    form_id = form["formId"]

    # If description is provided, add it using batchUpdate
    if description:
        service.forms().batchUpdate(
            formId=form_id,
            body={
                "requests": [
                    {
                        "updateFormInfo": {
                            "info": {"description": description},
                            "updateMask": "description",
                        }
                    }
                ]
            },
        ).execute()

    return form_id, form


def upload_image_to_drive_and_get_url(drive_service, image_path):
    """Upload image to Google Drive and return shareable URL."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    filename = os.path.basename(image_path)

    # Upload file to Drive
    media = MediaFileUpload(image_path, resumable=True)
    file_metadata = {"name": f"meme_{filename}", "parents": []}  # Upload to root folder

    file = drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()

    file_id = file.get("id")

    # Make the file publicly viewable
    drive_service.permissions().create(
        fileId=file_id, body={"role": "reader", "type": "anyone"}
    ).execute()

    # Return the direct link to the image
    return f"https://drive.google.com/uc?id={file_id}"


def create_image_item_with_url(image_url):
    """Create an image item using Google Drive URL."""
    return {
        "item": {
            "title": "",
            "imageItem": {
                "image": {
                    "sourceUri": image_url,
                    "properties": {"width": 400, "alignment": "CENTER"},
                }
            },
        }
    }


def create_text_question(config: TextQuestionConfig):
    """Create a text question item."""
    return {
        "item": {
            "title": config.text,
            "description": config.description,
            "questionItem": {"question": {"textQuestion": {}, "required": config.mandatory}},
        }
    }


def create_paragraph_question(config: ParagraphQuestionConfig):
    """Create a paragraph text question item."""
    return {
        "item": {
            "title": config.text,
            "description": config.description,
            "questionItem": {
                "question": {"textQuestion": {"paragraph": True}, "required": config.mandatory}
            },
        }
    }


def create_choice_question(config: ChoiceQuestionConfig):
    """Create a multiple choice question item."""
    question_type = "DROP_DOWN" if config.dropdown else "RADIO"
    return {
        "item": {
            "title": config.text,
            "description": config.description,
            "questionItem": {
                "question": {
                    "choiceQuestion": {
                        "type": question_type,
                        "options": [{"value": option} for option in config.choices or []],
                    },
                    "required": config.mandatory,
                }
            },
        }
    }


def create_scale_question(config: ScaleQuestionConfig):
    """Create a generic linear scale question."""
    scale_question_payload = {
        "low": config.low,
        "high": config.high,
    }
    if config.low_label is not None:
        scale_question_payload["lowLabel"] = config.low_label
    if config.high_label is not None:
        scale_question_payload["highLabel"] = config.high_label

    return {
        "item": {
            "title": config.text,
            "description": config.description,
            "questionItem": {
                "question": {"scaleQuestion": scale_question_payload, "required": config.mandatory}
            },
        }
    }


def add_questions_to_form(service, form_id, questions):
    """Add a list of questions to a form."""
    requests = []
    for index, question in enumerate(questions):
        question["location"] = {"index": index}
        requests.append({"createItem": question})

    if requests:
        service.forms().batchUpdate(formId=form_id, body={"requests": requests}).execute()


def create_question_from_config(question_config: AnyQuestionConfig):
    """Create a question item from configuration."""

    dispatch = {
        "text": create_text_question,
        "paragraph": create_paragraph_question,
        "choice": create_choice_question,
        "scale": create_scale_question,
    }

    return dispatch[question_config.kind](question_config)
