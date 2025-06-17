from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from googleapiclient.discovery import build

from utils.google_utils import SimpleGoogleAPI


@dataclass
class FormItem:
    """Represents a question or section header in a Google Form."""

    item_id: str
    title: str
    type: str
    index: int
    description: Optional[str] = None

    @classmethod
    def from_api_response(cls, item: Dict[str, Any], index: int) -> "FormItem":
        """Create a FormItem from the Google Forms API response."""
        item_id = item.get("itemId", "")
        title = ""
        item_type = ""
        description = None

        # Handle different item types
        if "title" in item:
            title = item["title"]
            item_type = "SECTION_HEADER"
            if "description" in item:
                description = item["description"]
        elif "questionItem" in item:
            question = item["questionItem"]["question"]
            title = question.get("questionId", "")
            if "title" in question:
                title = question["title"]
            item_type = question.get("type", "")
            if "questionParagraph" in question:
                item_type = "PARAGRAPH"
            elif "questionScale" in question:
                item_type = "SCALE"
            description = question.get("description", None)

        return cls(
            item_id=item_id, title=title, type=item_type, index=index, description=description
        )


class GoogleFormsAPI(SimpleGoogleAPI):
    """Extension of SimpleGoogleAPI to work with Google Forms."""

    def __init__(self, service_account_file: str):
        super().__init__(service_account_file)
        self.forms_service = build("forms", "v1", credentials=self.credentials)

    def get_form(self, form_id: str) -> Dict[str, Any]:
        """Get a Google Form by ID."""
        form = self.forms_service.forms().get(formId=form_id).execute()
        return form

    def get_form_items(self, form_id: str) -> List[FormItem]:
        """Get all items (questions, sections, etc.) in a Google Form."""
        form = self.get_form(form_id)
        items = []

        if "items" in form:
            for i, item in enumerate(form["items"]):
                items.append(FormItem.from_api_response(item, i))

        return items

    def find_sections_range(
        self, form_id: str, start_section: str, end_section: str
    ) -> Tuple[Optional[int], Optional[int]]:
        """Find the range of items between two section headers."""
        items = self.get_form_items(form_id)

        start_idx = None
        end_idx = None

        for i, item in enumerate(items):
            if item.type == "SECTION_HEADER":
                if item.title == start_section:
                    start_idx = i
                elif item.title == end_section and start_idx is not None:
                    end_idx = i
                    break

        return start_idx, end_idx

    def update_form(self, form_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a Google Form with the provided updates."""
        result = (
            self.forms_service.forms()
            .batchUpdate(formId=form_id, body={"requests": updates})
            .execute()
        )
        return result

    def delete_items(self, form_id: str, item_ids: List[str]) -> Dict[str, Any]:
        """Delete items from a form by their IDs."""
        requests = [{"deleteItem": {"itemId": item_id}} for item_id in item_ids]
        return self.update_form(form_id, requests)

    def create_scale_question(
        self,
        form_id: str,
        title: str,
        description: str = None,
        lower: int = 1,
        upper: int = 5,
        lower_label: str = "Poor",
        upper_label: str = "Excellent",
    ) -> Dict[str, Any]:
        """Create a scale question in the form."""
        request = {
            "createItem": {
                "item": {
                    "title": title,
                    "questionItem": {
                        "question": {
                            "required": False,
                            "scaleQuestion": {
                                "low": lower,
                                "high": upper,
                                "lowLabel": lower_label,
                                "highLabel": upper_label,
                            },
                        }
                    },
                },
                "location": {"index": 0},  # Will be placed at the end by default
            }
        }

        if description:
            request["createItem"]["item"]["description"] = description

        return (
            self.forms_service.forms()
            .batchUpdate(formId=form_id, body={"requests": [request]})
            .execute()
        )

    def create_paragraph_question(
        self, form_id: str, title: str, description: str = None
    ) -> Dict[str, Any]:
        """Create a paragraph question in the form."""
        request = {
            "createItem": {
                "item": {
                    "title": title,
                    "questionItem": {"question": {"required": False, "paragraphQuestion": {}}},
                },
                "location": {"index": 0},  # Will be placed at the end by default
            }
        }

        if description:
            request["createItem"]["item"]["description"] = description

        return (
            self.forms_service.forms()
            .batchUpdate(formId=form_id, body={"requests": [request]})
            .execute()
        )
