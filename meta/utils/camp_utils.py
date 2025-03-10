import csv
import datetime
from pathlib import Path
import re
import string
from pydantic import BaseModel
import random

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from utils.openai_utils import ServiceAccount

CAMPS_DIR = Path(__file__).parent.parent / "camps"

PATTERN_TEAMUP_URL = re.compile("https://teamup.com/([a-z0-9]+)")


class Camp(BaseModel):
    name: str
    password: str
    date: str
    teamup_admin_url: str | None = None
    participants_name_and_email_csv: str = "name,email\n"
    openai_camp_service_account: ServiceAccount | None = None
    # If you add a new field here, remember to add it to edit_camp.py too.

    @classmethod
    def new(cls, name: str, date: str) -> "Camp":
        password = "".join(random.choices(string.ascii_letters, k=16))
        return cls(name=name, password=password, date=date, teamup_admin_url=None)

    def save_to_disk(self):
        campfile = CAMPS_DIR / f"{self.name}.json"
        campfile.write_text(self.model_dump_json())

    @classmethod
    def load_from_disk(self, name: str) -> "Camp":
        assert "\\" not in name, "Invalid camp name"
        path = CAMPS_DIR / f"{name}.json"
        camp = self.model_validate_json(path.read_text("utf-8"))
        assert camp.name == name, f"Camp name {camp.name} does not match file name {name}"
        return camp

    def validate_teamup(self) -> str | None:
        if not self.teamup_admin_url:
            return "No teamup admin URL set"

        if not PATTERN_TEAMUP_URL.match(self.teamup_admin_url):
            return f"Invalid teamup admin URL. Must match `{PATTERN_TEAMUP_URL}`"

        return None

    @property
    def teamup_admin_calendar_key(self) -> str:
        return PATTERN_TEAMUP_URL.match(self.teamup_admin_url).group(1)

    @property
    def start_datetime(self):
        return datetime.datetime.fromisoformat(self.date)

    @staticmethod
    def list_all():
        camps = []
        for file in CAMPS_DIR.glob("*.json"):
            camp = Camp.model_validate_json(file.read_text("utf-8"))
            assert (
                camp.name == file.stem
            ), f"Camp name {camp.name} does not match file name {file.stem}"
            camps.append(camp)

        return camps

    def participants_list(self) -> list[str]:
        reader = csv.DictReader(self.participants_name_and_email_csv.splitlines())
        return list(row["name"] for row in reader)


def is_in_streamlit() -> bool:
    return get_script_run_ctx(suppress_warning=True) is not None


def get_current_camp() -> Camp | None:
    if is_in_streamlit():
        return st.session_state.get("current_camp", None)
    else:
        # Take the most recent one
        camps = Camp.list_all()
        if camps:
            return max(camps, key=lambda c: c.date)
        return None


def edit_current_camp(**kwargs):
    camp = get_current_camp()
    if not camp:
        raise ValueError("No camp selected")

    new_camp = camp.model_copy(update=kwargs)

    new_camp.save_to_disk()
    st.session_state.current_camp = camp
