import json

import streamlit as st
from streamlit_session_browser_storage import SessionStorage

from utils.camp_utils import Camp


class State:
    CAMP_PASSWORDS_KEY = "camp_passwords"
    CURRENT_CAMP_KEY = "current_camp"

    def __init__(self):
        self.container = st.sidebar.container(height=0)
        with self.container:
            # Hides self.container completely, and extra elements created by SessionStorage
            st.markdown(
                """
<style class="hide-parent">
    div:has(> div > div > div > div > div > .hide-parent) {
        display: none;
    }
</style>
""",
                unsafe_allow_html=True,
            )
            self._session_storage = SessionStorage()

    def login(self, camp_name: str, password: str, save_to_browser: bool = True) -> Camp | None:
        camp = Camp.load_from_disk(camp_name)
        if camp.password != password:
            return None

        st.session_state.current_camp = camp
        if save_to_browser:
            self.save_camp_password_in_browser(camp.name, password)
            self.save_camp_in_browser(camp.name)
        return camp

    def auto_login(self, camp_name: str | None = None) -> Camp | None:
        if camp_name is None:
            # Try to use the last camp from the browser
            camp_name = self.get_last_campname_from_browser()
            if camp_name is None:
                return None

        camp = Camp.load_from_disk(camp_name)
        known_passwords = self.get_camp_passwords()
        return self.login(camp_name, known_passwords.get(camp.name, ""), save_to_browser=False)

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
            self._session_storage.setItem(self.CURRENT_CAMP_KEY, camp, key="set-camp")

    @property
    def current_camp(self) -> Camp | None:
        return st.session_state.get(self.CURRENT_CAMP_KEY, None)

    # ----

    def login_form(self, key: str = "login_form") -> Camp | None:
        camps = Camp.list_all()
        camps.sort(key=lambda c: c.date, reverse=True)

        if not camps:
            st.warning("No camps found. Please create a camp first.")
            return None

        default_camp = self.get_last_campname_from_browser()
        default_camp_idx = next((i for i, c in enumerate(camps) if c.name == default_camp), 0)
        camp: Camp = st.selectbox(
            "Select camp",
            camps,
            format_func=lambda c: c.name,
            index=default_camp_idx,
            key=key + "_camp_select",
        )
        assert camp is not None

        if self.auto_login(camp.name):
            st.write(f"You are logged in for `{camp.name}`.")
            if st.button("Log out", on_click=self.logout, key=key + "_logout"):
                st.toast("You were logged out.")
                return None
            return camp

        password = st.text_input(
            "Password",
            type="password",
            key=key + "_password",
        )

        if not password:
            return None

        if self.login(camp.name, password):
            st.success("You were logged in.")
            return camp
        else:
            st.error("Invalid password")
            return None
