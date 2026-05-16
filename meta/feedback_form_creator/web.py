import streamlit as st
import yaml
from googleapiclient.errors import HttpError
from pydantic import ValidationError
from streamlit_ace import st_ace

from meta.feedback_form_creator.cli import create_daily_feedback_form
from meta.feedback_form_creator.forms_utils import (
    get_drive_service,
    get_forms_service,
)
from meta.feedback_form_creator.models import CampConfig, DayConfig
from meta.shared.google import service_account_email
from meta.web.helpers import (
    State,
    edit_current_camp,
    get_current_camp,
    render_select_camp_message,
)

REPO = "https://github.com/EffiSciencesResearch/ML4G-2.0/blob/master/meta/feedback_form_creator"

state = State()

with st.sidebar:
    state.login_form()

st.title("Feedback Form Creator")

camp = get_current_camp()
if not camp:
    render_select_camp_message()
    st.stop()

st.write(f"Editing feedback config for camp `{camp.name}`.")

# --- Config editor -----------------------------------------------------------

st.subheader("Config")
st.write(
    "Edit the YAML config for this camp's daily feedback forms. "
    f"`camp_name` defaults to `{camp.name}` so you can omit it. "
    f"See the [schema]({REPO}/models.py) for what's possible, "
    f"and the [available memes]({REPO}/memes) to set per day."
)

yaml_text = st_ace(
    value=camp.feedback_config_yaml,
    language="yaml",
    theme="github",
    keybinding="vscode",
    show_gutter=True,
    auto_update=True,
    height=1000,
    key="feedback_yaml",
)


def _parse(text: str) -> CampConfig:
    raw = yaml.safe_load(text) or {}
    raw.setdefault("camp_name", camp.name)
    return CampConfig.model_validate(raw)


parsed: CampConfig | None = None
parse_error: str | None = None
if yaml_text.strip():
    try:
        parsed = _parse(yaml_text)
    except (yaml.YAMLError, ValidationError) as e:
        parse_error = str(e)

if parse_error:
    st.error(f"Config is not valid:\n\n```\n{parse_error}\n```")
elif parsed:
    st.success(f"Config parses cleanly — {len(parsed.timetable)} day(s) defined.")

cols = st.columns([1, 1, 4])
with cols[0]:
    save_disabled = parse_error is not None or yaml_text == camp.feedback_config_yaml
    if st.button("Save", type="primary", disabled=save_disabled):
        edit_current_camp(feedback_config_yaml=yaml_text)
        st.success("Saved to camp.")
        st.rerun()
with cols[1]:
    if st.button("Revert", disabled=yaml_text == camp.feedback_config_yaml):
        st.rerun()

# --- Form creator ------------------------------------------------------------

st.subheader("Create a form")

if not parsed:
    st.info("Save a valid config above to enable form creation.")
    st.stop()

day_names = list(parsed.timetable.keys())
day_name = st.pills("Day", options=day_names, default=day_names[0])
if not day_name:
    st.stop()
day_number = day_names.index(day_name) + 1
day_config = parsed.timetable[day_name]


def _preview_question_titles(config: CampConfig, day: DayConfig) -> list[str]:
    """Mirrors create_daily_feedback_form to list the question titles that would be created."""
    titles = [q.text for q in config.pre_questions]
    for s in day.sessions:
        titles.append(f"How would you rate the '{s.name}' session?")
        if s.reading_group:
            titles.append(f"Which teacher facilitated the '{s.name}' reading group?")
        titles.append(f"Any additional feedback on '{s.name}'?")
    titles.extend(q.text for q in day.day_questions)
    titles.extend(q.text for q in config.post_questions)
    if day.meme:
        titles.append("Provide a caption for this meme that describes your day!")
    return titles


titles = _preview_question_titles(parsed, day_config)
with st.expander(f"Preview: {len(titles)} questions would be created", expanded=True):
    for t in titles:
        st.markdown(f"- {t}")


def _render_folder_access_error(folder_id: str, error: HttpError) -> None:
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    if error.resp.status == 403:
        bot = service_account_email()
        if bot:
            st.error(
                f"The bot can't access this Drive folder. Give **Editor** rights "
                f"to `{bot}` on [the folder]({folder_url}) and try again."
            )
            return
    st.error(f"Cannot access Drive folder [{folder_id}]({folder_url}): {error}")


if st.button(f"Create form for {day_name}", type="primary"):
    with st.spinner("Authenticating with Google..."):
        forms_service = get_forms_service()
        drive_service = get_drive_service()

    if parsed.drive_folder_id:
        try:
            drive_service.files().get(
                fileId=parsed.drive_folder_id, fields="id", supportsAllDrives=True
            ).execute()
        except HttpError as e:
            _render_folder_access_error(parsed.drive_folder_id, e)
            st.stop()

    with st.spinner(f"Creating form for {day_name}..."):
        form_id, form = create_daily_feedback_form(
            forms_service, drive_service, parsed, day_name, day_number, day_config
        )

    st.success("Form created.")
    st.markdown(
        f"- **Edit:** https://docs.google.com/forms/d/{form_id}/edit\n"
        f"- **Live:** {form['responderUri']}"
    )
