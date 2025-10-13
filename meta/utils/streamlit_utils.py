from typing import Annotated

from pydantic import BaseModel, Field
import streamlit as st
from streamlit_local_storage import LocalStorage

from utils.camp_utils import Camp


class PerBrowserSettings(BaseModel):
    camp_passwords: Annotated[dict[str, str], Field(default_factory=dict)]
    current_camp: str | None = None
    dashboard_name: str | None = None

    class Config:
        frozen = True


class State:

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
            self._browser_local_storage = LocalStorage()

        self.per_browser_settings = self.load_settings()

    def load_settings(self) -> PerBrowserSettings:
        stored = self._browser_local_storage.getItem("ml4g-settings")
        if stored is not None:
            return PerBrowserSettings.model_validate(stored)
        return PerBrowserSettings()

    def save_settings(self, **kwargs):
        # Check if it changed
        if all(getattr(self.per_browser_settings, k) == v for k, v in kwargs.items()):
            return
        new = self.per_browser_settings.model_copy(update=kwargs)
        with self.container:
            self._browser_local_storage.setItem("ml4g-settings", new.model_dump())
        self.per_browser_settings = new

    def login(self, camp_name: str, password: str, save_to_browser: bool = True) -> Camp | None:
        camp = Camp.load_from_disk(camp_name)
        if camp.password != password:
            return None

        st.session_state.current_camp = camp
        if save_to_browser:
            self.save_settings(
                current_camp=camp.name,
                camp_passwords={**self.per_browser_settings.camp_passwords, camp.name: password},
            )
        return camp

    def auto_login(self, camp_name: str | None = None) -> Camp | None:
        if camp_name is None:
            # Try to use the last camp from the browser
            camp_name = self.per_browser_settings.current_camp
            if camp_name is None:
                return None

        camp = Camp.load_from_disk(camp_name)
        return self.login(
            camp_name,
            self.per_browser_settings.camp_passwords.get(camp.name, ""),
            save_to_browser=False,
        )

    def logout(self):
        st.session_state.pop("current_camp", None)
        with self.container:
            self.save_settings(camp_passwords={})

    def select_camp(self, name: str):
        camp = Camp.load_from_disk(name)
        st.session_state.current_camp = camp
        self.save_settings(current_camp=camp.name)

    @property
    def current_camp(self) -> Camp | None:
        return st.session_state.get("current_camp", None)

    # ----

    def login_form(self, key: str = "login_form") -> Camp | None:
        camps = Camp.list_all()
        camps.sort(key=lambda c: c.date, reverse=True)

        if not camps:
            st.warning("No camps found. Please create a camp first.")
            return None

        default_camp = self.per_browser_settings.current_camp
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


def render_select_camp_message():
    st.write("## 👈 Please select a camp on the left sidebar to continue.")
    st.write(
        "On a small screen the sidebar might be collapsed, but you'll find a `>` button to expand it on the top right. "
        "If you're an organiser/teacher/TA, a password should have been given to you by the main camp organizer, if not, contact them."
    )
