import json

import streamlit as st
from streamlit_session_browser_storage import SessionStorage

from camp_utils import Camp


class State:
    CAMP_PASSWORDS_KEY = "camp_passwords"
    CURRENT_CAMP_KEY = "current_camp"

    def __init__(self):
        self.container = st.sidebar.empty()
        with self.container:
            self._session_storage = SessionStorage()

    def login(self, camp_name: str, password: str) -> bool:
        camp = Camp.load_from_disk(camp_name)
        if camp.password != password:
            return False

        st.session_state.current_camp = camp
        self.save_camp_password_in_browser(camp.name, password)
        return True

    def auto_login(self, camp_name: str | None = None) -> bool:
        if camp_name is None:
            # Try to use the last camp from the browser
            camp_name = self.get_last_campname_from_browser()
            if camp_name is None:
                return False

        camp = Camp.load_from_disk(camp_name)
        known_passwords = self.get_camp_passwords()
        return self.login(camp_name, known_passwords.get(camp.name, ""))

    def logout(self):
        st.session_state.pop("current_camp", None)
        with self.container:
            self._session_storage.deleteItem(self.CAMP_PASSWORDS_KEY)
            # self._session_storage.deleteItem(self.CURRENT_CAMP_KEY)  # not needed

    def select_camp(self, name: str):
        camp = Camp.load_from_disk(name)
        st.session_state.current_camp = camp
        self.save_camp_in_browser(camp.name)

    def get_camp_passwords(self) -> dict[str, str]:
        data = self._session_storage.getItem(self.CAMP_PASSWORDS_KEY)
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return {}

    def get_last_campname_from_browser(self) -> str | None:
        return self._session_storage.getItem(self.CURRENT_CAMP_KEY)

    def save_camp_password_in_browser(self, name: str, password: str):
        passwords = self.get_camp_passwords()
        passwords[name] = password
        with self.container:
            self._session_storage.setItem(self.CAMP_PASSWORDS_KEY, json.dumps(passwords))

    def save_camp_in_browser(self, camp: str):
        with self.container:
            self._session_storage.setItem(self.CURRENT_CAMP_KEY, camp)

    @property
    def current_camp(self) -> Camp | None:
        return st.session_state.get(self.CURRENT_CAMP_KEY, None)
