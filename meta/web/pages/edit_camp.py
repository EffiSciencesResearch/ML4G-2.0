import csv
import datetime

import streamlit as st

from meta.web.openrouter import CAMP_KEY_DURATION_DAYS, CAMP_KEY_LIMIT_USD, OpenRouterAPIKey
from meta.web.helpers import (
    State,
    edit_current_camp,
    get_current_camp,
    render_select_camp_message,
)

state = State()

with st.sidebar:
    state.login_form()


st.title("Edit Camp details")
camp = get_current_camp()
if not camp:
    render_select_camp_message()
    st.stop()

st.write(f"## Currently editing camp `{camp.name}`")

top_save_button = st.button("Save", key="top-save", type="primary")

date = st.date_input("Date", value=datetime.datetime.fromisoformat(camp.date))

# Password: read only
password = st.text_input("Password", disabled=True, value=camp.password)

# --------------------------------
st.subheader("OpenRouter API key")
st.write(
    f"We use an API key for some of the notebooks in the camp. "
    f"You can create one here, and don't forget to revoke it after the camp.\n\n"
    f"Each key has a **\\${CAMP_KEY_LIMIT_USD:.0f} lifetime credit limit** that does not renew, "
    f"and automatically **expires after {CAMP_KEY_DURATION_DAYS} days**. "
    f"This is generally enough for one camp — we mostly use the cheaper models to demo usage. "
    f"If we run out, you can deactivate and recreate another key. "
    f"If more credits are needed, contact Diego or Nia."
)
if camp.openrouter_api_key is None:

    def new_openrouter_key():
        edit_current_camp(openrouter_api_key=OpenRouterAPIKey.from_name(camp.name))

    st.button("Create OpenRouter API key", on_click=new_openrouter_key)
else:

    def delete_openrouter_key():
        camp.openrouter_api_key.delete()
        edit_current_camp(openrouter_api_key=None)

    st.write(f"API key:\n```\n{camp.openrouter_api_key.api_key}\n```")
    st.write(f"Credit limit: **\\${camp.openrouter_api_key.limit:.2f}** (does not renew)")
    if camp.openrouter_api_key.expires_at is not None:
        st.write(f"Expires: **{camp.openrouter_api_key.expires_at.strftime('%Y-%m-%d')}**")
    st.button("Revoke OpenRouter API key", on_click=delete_openrouter_key)

    if st.button("Check usage"):
        try:
            usage = camp.openrouter_api_key.fetch_usage()
            spent = usage.get("usage", 0)
            limit = usage.get("limit") or camp.openrouter_api_key.limit
            st.write(f"Spent: **\\${spent:.4f}** of **\\${limit:.2f}**")
        except Exception as e:
            st.error(f"Could not fetch usage: {e}")

st.subheader("Feedback Sheet")
st.write(
    "The feedback sheet URL allows us to analyze session feedback from your Google Sheets. "
    "Make sure the service account has read access to the sheet."
)
feedback_sheet_url = st.text_input(
    "Feedback sheet URL",
    value=camp.feedback_sheet_url or "",
    placeholder="https://docs.google.com/spreadsheets/d/...",
    help="Google Sheets URL containing session feedback data",
)

st.subheader("Participants")
st.write(
    "We need the participants names and emails here mostly to send the career planning docs and pre-fill the 1-1 schedule. "
    "We put it as CSV to make it easier to copy-paste and edit."
)
participants = st.text_area(
    "Participants name and email CSV",
    value=camp.participants_name_and_email_csv,
    placeholder="name,email\njack,jack@ml4good.org\njulia,julia@ml4bad.com",
    help="CSV with columns 'name' and 'email'.",
    height=200,
)

# Check that there's at least name & email columns
detected_columns = csv.DictReader(participants.splitlines()).fieldnames
if not detected_columns or not all(col in detected_columns for col in ["name", "email"]):
    st.error(
        f"""The CSV should have headers `name` and `email` in the first line. Example:\n
```
name,email
jack,jack@ml4good.org
julia,julia@ml4bad.com
```
Detected columns: {detected_columns}
        """
    )
    st.stop()

if st.button("Save", type="primary") or top_save_button:
    new_data = dict(
        date=date.isoformat(),
        participants_name_and_email_csv=participants,
        feedback_sheet_url=feedback_sheet_url,
    )

    # Remove newdata that is None
    new_data = {k: v for k, v in new_data.items() if v not in (None, "")}
    edit_current_camp(**new_data)
