import json
from pathlib import Path

import streamlit as st
from streamlit_session_browser_storage import SessionStorage

from camp_utils import Camp, list_camps


class State:
    CAMP_PASSWORDS_KEY = "camp_passwords"

    def __init__(self):
        self.container = st.sidebar.empty()
        with self.container:
            self._session_storage = SessionStorage()

    def login(self, camp_file: Path, password: str) -> bool:
        camp = self.load_camp(camp_file)
        if camp.password != password:
            return False

        st.session_state.current_camp = camp
        st.session_state.current_campfile = camp_file
        self.save_camp_password_in_browser(camp.name, password)
        return True

    def auto_login(self, camp_file: Path | None = None) -> bool:
        if camp_file is None:
            # Try to use the last camp from the browser
            camp_file = self.get_last_campfile_from_browser()
            if camp_file is None:
                return False

        camp = self.load_camp(camp_file)
        known_passwords = self.get_camp_passwords()
        return self.login(camp_file, known_passwords.get(camp.name, ""))

    def logout(self):
        st.session_state.pop("current_camp", None)
        st.session_state.pop("current_campfile", None)
        with self.container:
            self._session_storage.deleteItem(self.CAMP_PASSWORDS_KEY)

    def select_camp(self, camp_file: Path):
        camp = self.load_camp(camp_file)
        st.session_state.current_camp = camp
        st.session_state.current_campfile = camp_file
        self.save_camp_in_browser(camp.name)

    def get_camp_passwords(self) -> dict[str, str]:
        data = self._session_storage.getItem(self.CAMP_PASSWORDS_KEY)
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return {}

    def get_last_campfile_from_browser(self) -> Path | None:
        camp_name = self._session_storage.getItem("current_camp")
        if not camp_name:
            return None

        camps = list_camps()
        for camp_file, camp in camps.items():
            if camp.name == camp_name:
                return camp_file

        return None

    def load_camp(self, camp_file: Path) -> Camp:
        return Camp.model_validate_json(camp_file.read_text("utf-8"))

    def save_camp_password_in_browser(self, camp_name: str, password: str):
        passwords = self.get_camp_passwords()
        passwords[camp_name] = password
        with self.container:
            self._session_storage.setItem(self.CAMP_PASSWORDS_KEY, json.dumps(passwords))

    def save_camp_in_browser(self, camp: str):
        with self.container:
            self._session_storage.setItem("current_camp", camp)

    @property
    def current_camp(self) -> Camp | None:
        return st.session_state.get("current_camp", None)

    @property
    def current_campfile(self) -> Path | None:
        return st.session_state.get("current_campfile", None)
