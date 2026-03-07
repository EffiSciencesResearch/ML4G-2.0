from typing import Annotated

from pydantic import BaseModel, Field
import streamlit as st
from streamlit_local_storage import LocalStorage

from utils.camp_utils import Camp
from utils.storage import Storage, get_storage as _get_storage


@st.cache_resource
def get_storage() -> Storage:
    return _get_storage()


def get_current_camp() -> Camp | None:
    return st.session_state.get("current_camp", None)


def edit_current_camp(**kwargs):
    camp = get_current_camp()
    if not camp:
        raise ValueError("No camp selected")
    new_camp = camp.model_copy(update=kwargs)
    get_storage().save_camp(new_camp)
    st.session_state.current_camp = new_camp


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
        camp = get_storage().load_camp(camp_name)
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

        saved_password = self.per_browser_settings.camp_passwords.get(camp_name, "")
        return self.login(camp_name, saved_password, save_to_browser=False)

    def logout(self):
        st.session_state.pop("current_camp", None)
        with self.container:
            self.save_settings(camp_passwords={})

    def select_camp(self, name: str):
        camp = get_storage().load_camp(name)
        st.session_state.current_camp = camp
        self.save_settings(current_camp=camp.name)

    @property
    def current_camp(self) -> Camp | None:
        return st.session_state.get("current_camp", None)

    # ----

    def login_form(self, key: str = "login_form") -> Camp | None:
        camp_names = get_storage().list_camps()

        if not camp_names:
            st.warning("No camps found. Please create a camp first.")
            return None

        default_camp = self.per_browser_settings.current_camp
        default_idx = next((i for i, n in enumerate(camp_names) if n == default_camp), 0)
        selected_name: str = st.selectbox(
            "Select camp",
            camp_names,
            index=default_idx,
            key=key + "_camp_select",
        )
        assert selected_name is not None

        if self.auto_login(selected_name):
            st.write(f"You are logged in for `{selected_name}`.")
            if st.button("Log out", on_click=self.logout, key=key + "_logout"):
                st.toast("You were logged out.")
                return None
            return self.current_camp

        password = st.text_input(
            "Password",
            type="password",
            key=key + "_password",
        )

        if not password:
            return None

        camp = self.login(selected_name, password)
        if camp:
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
