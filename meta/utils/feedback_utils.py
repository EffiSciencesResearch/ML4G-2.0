import re
from dataclasses import dataclass

import pandas as pd

from utils.google_utils import SimpleGoogleAPI, extract_id_from_url


@dataclass
class FeedbackEntry:
    """A single feedback entry for a session."""

    session_name: str
    participant_name: str | None
    rating: int | None  # 1-5 scale
    feedback_text: str | None
    tab_source: str  # Which sheet tab this came from


@dataclass
class SessionFeedback:
    """Aggregated feedback for a single session."""

    session_name: str
    entries: list[FeedbackEntry]

    @property
    def ratings(self) -> list[int]:
        """Get all valid ratings for this session."""
        return [entry.rating for entry in self.entries if entry.rating is not None]

    @property
    def feedback_texts(self) -> list[str]:
        """Get all non-empty feedback texts for this session."""
        return [
            entry.feedback_text
            for entry in self.entries
            if entry.feedback_text and entry.feedback_text.strip()
        ]

    @property
    def average_rating(self) -> float | None:
        """Calculate average rating for this session."""
        ratings = self.ratings
        return sum(ratings) / len(ratings) if ratings else None


class FeedbackParser:
    """Parser for extracting session feedback from Google Sheets."""

    def __init__(self, api: SimpleGoogleAPI):
        self.api = api

    def parse_feedback_sheet(self, sheet_url: str) -> dict[str, SessionFeedback]:
        """Parse all feedback from a Google Sheet URL."""
        sheet_id = extract_id_from_url(sheet_url)
        all_sheets_data = self.api.get_all_sheets_data(sheet_id)

        # Parse each sheet and aggregate feedback
        all_feedback: dict[str, list[FeedbackEntry]] = {}

        for tab_name, raw_data in all_sheets_data.items():
            if not raw_data:  # Skip empty sheets
                continue

            entries = self._parse_sheet_data(raw_data, tab_name)

            # Group entries by session
            for entry in entries:
                if entry.session_name not in all_feedback:
                    all_feedback[entry.session_name] = []
                all_feedback[entry.session_name].append(entry)

        # Convert to SessionFeedback objects
        return {
            session_name: SessionFeedback(session_name, entries)
            for session_name, entries in all_feedback.items()
        }

    def _parse_sheet_data(self, raw_data: list[list[str]], tab_name: str) -> list[FeedbackEntry]:
        """Parse raw sheet data to extract feedback entries."""
        if not raw_data or len(raw_data) < 2:
            return []

        # Convert to DataFrame for easier processing (first row is always headers)
        df = pd.DataFrame(raw_data[1:], columns=raw_data[0])

        # Find columns that contain session ratings and feedback
        rating_columns = self._find_rating_columns(df.columns)
        feedback_columns = self._find_feedback_columns(df.columns)

        # Find name column
        name_column = self._find_name_column(df.columns)

        entries = []

        for _, row in df.iterrows():
            participant_name = row.get(name_column) if name_column else None

            # Process each session found in the columns
            for session_name in rating_columns:
                rating_col = rating_columns[session_name]
                feedback_col = feedback_columns.get(session_name)

                # Extract rating (convert to int if possible)
                rating = None
                if rating_col in row and pd.notna(row[rating_col]):
                    try:
                        rating = int(float(row[rating_col]))
                        if not (1 <= rating <= 5):  # Validate rating range
                            rating = None
                    except (ValueError, TypeError):
                        rating = None

                # Extract feedback text
                feedback_text = None
                if feedback_col and feedback_col in row and pd.notna(row[feedback_col]):
                    feedback_text = str(row[feedback_col]).strip()
                    if not feedback_text:
                        feedback_text = None

                # Only create entry if we have either rating or feedback
                if rating is not None or feedback_text is not None:
                    entries.append(
                        FeedbackEntry(
                            session_name=session_name,
                            participant_name=participant_name,
                            rating=rating,
                            feedback_text=feedback_text,
                            tab_source=tab_name,
                        )
                    )

        return entries

    def _find_rating_columns(self, columns: list[str]) -> dict[str, str]:
        """Find columns that contain session ratings."""
        rating_columns = {}

        # Simplified pattern - assumes all rating columns have quotes around session names
        pattern = re.compile(r"How would you rate the '([^']+)'")
        #   - 'How would you rate the 'Intro to AI Safety' afternoon session?'
        #  - 'How would you rate the 'Opening session'?'
        for col in columns:
            match = pattern.search(col)
            if match:
                session_name = match.group(1).strip()
                rating_columns[session_name] = col

        return rating_columns

    def _find_feedback_columns(self, columns: list[str]) -> dict[str, str]:
        """Find columns that contain session feedback text."""
        feedback_columns = {}

        # Simplified pattern - assumes all feedback columns have quotes around session names
        pattern = re.compile(r"Any additional feedback on '([^']+)'")

        for col in columns:
            match = pattern.search(col)
            if match:
                session_name = match.group(1).strip()
                feedback_columns[session_name] = col

        return feedback_columns

    def _find_name_column(self, columns: list[str]) -> str | None:
        """Find the column that contains participant names."""
        name_patterns = [
            r"^names?$",
            r"^participant\s*names?$",
            r"^full\s*names?$",
            r"^your\s*names?$",
        ]

        for col in columns:
            for pattern in name_patterns:
                if re.match(pattern, col, re.IGNORECASE):
                    return col

        return None
