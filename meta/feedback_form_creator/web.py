import streamlit as st
import yaml
from pydantic import ValidationError

from meta.feedback_form_creator.forms_utils import (
    get_drive_service,
    get_forms_service,
)
from meta.feedback_form_creator.models import CampConfig
from meta.feedback_form_creator.cli import create_daily_feedback_form
from meta.web.helpers import (
    State,
    edit_current_camp,
    get_current_camp,
    render_select_camp_message,
)

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
st.caption(
    "Paste or edit the YAML config for this camp's daily feedback forms. "
    "Schema is the same as `meta/feedback_form_creator/config.yaml`; the `camp_name` field "
    f"is taken from the camp ({camp.name}) so you can omit it here."
)

yaml_text = st.text_area(
    label="config.yaml",
    value=camp.feedback_config_yaml,
    height=500,
    key="feedback_yaml",
    label_visibility="collapsed",
)


def _parse(text: str) -> CampConfig | None:
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
day_name = st.selectbox("Day", day_names)
day_number = day_names.index(day_name) + 1
day_config = parsed.timetable[day_name]

st.write(f"**{len(day_config.sessions)}** session(s) on `{day_name}`.")

if st.button(f"Create form for {day_name}", type="primary"):
    with st.spinner("Authenticating with Google..."):
        forms_service = get_forms_service()
        drive_service = get_drive_service()

    if parsed.drive_folder_id:
        try:
            drive_service.files().get(
                fileId=parsed.drive_folder_id, fields="id", supportsAllDrives=True
            ).execute()
        except Exception as e:
            st.error(f"Cannot access Drive folder `{parsed.drive_folder_id}`: {e}")
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
