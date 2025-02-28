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
    name: str
    password: str
    date: str
    teamup_admin_url: str | None

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
        camps[camp_file] = Camp.model_validate_json(camp_file.read_text())

    return camps


def is_in_streamlit() -> bool:
    return get_script_run_ctx(suppress_warning=True) is not None


def get_current_camp() -> Camp | None:
    if is_in_streamlit():
        return st.session_state.get("current_camp", None)
    else:
        # Take the last one
        camps = list_camps()
        if camps:
            return max(camps.values(), key=lambda c: c.date)
        return None


def set_current_camp(camp: Camp):
    st.session_state.current_camp = camp
