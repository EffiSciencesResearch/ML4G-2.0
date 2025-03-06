from pathlib import Path
import re
import string
from pydantic import BaseModel
import random

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

CAMPS_DIR = Path(__file__).parent / "camps"

PATTERN_TEAMUP_URL = re.compile("https://teamup.com/([a-z0-9]+)")


class Camp(BaseModel):
    # If you add a new field here, remember to add it to edit_camp.py too.
    name: str
    password: str
    date: str
    teamup_admin_url: str | None = None
    participants_name_and_email_csv: str = "name,email\n"

    @classmethod
    def new(cls, name: str, date: str) -> "Camp":
        password = "".join(random.choices(string.ascii_letters, k=16))
        return cls(name=name, password=password, date=date, teamup_admin_url=None)

    def validate_teamup(self) -> str | None:
        if not self.teamup_admin_url:
            return "No teamup admin URL set"

        if not PATTERN_TEAMUP_URL.match(self.teamup_admin_url):
            return f"Invalid teamup admin URL. Must match `{PATTERN_TEAMUP_URL}`"

        return None

    @property
    def teamup_admin_calendar_key(self) -> str:
        return PATTERN_TEAMUP_URL.match(self.teamup_admin_url).group(1)


def list_camps() -> dict[Path, Camp]:
    camps = {}
    for camp_file in CAMPS_DIR.glob("*.json"):
        camps[camp_file] = Camp.model_validate_json(camp_file.read_text("utf-8"))

    return camps


def is_in_streamlit() -> bool:
    return get_script_run_ctx(suppress_warning=True) is not None


def get_current_camp() -> Camp | None:
    if is_in_streamlit():
        return st.session_state.get("current_camp", None)
    else:
        campfile = get_current_campfile()
        if campfile:
            return list_camps().get(campfile, None)
        else:
            return None


def get_current_campfile() -> Path | None:
    if is_in_streamlit():
        return st.session_state.get("current_campfile", None)
    else:
        # Take the last one
        camps = list_camps()
        if camps:
            return max(camps.keys(), key=lambda c: camps[c].date)
        return None


def edit_current_camp(**kwargs):
    camp = get_current_camp()
    campfile = get_current_campfile()
    if not camp:
        raise ValueError("No camp selected")

    new_camp = camp.model_copy(update=kwargs)

    campfile.write_text(new_camp.model_dump_json())
    st.session_state.current_camp = camp
