import streamlit as st
from utils.openai_utils import ServiceAccount
from utils.camp_utils import get_current_camp, edit_current_camp
import datetime
from utils.streamlit_utils import State

state = State()

with st.sidebar:
    state.login_form()


camp = get_current_camp()
if not camp:
    st.error("Please select a camp first on the main page.")
    st.stop()

st.write(f"## Currently editing camp `{camp.name}`")

top_save_button = st.button("Save", key="top-save", type="primary")

date = st.date_input("Date", value=datetime.datetime.fromisoformat(camp.date))

st.subheader("TeamUp")
st.write(
    "The [TeamUp](https://teamup.com) admin URL allows to edit our calendars. "
    "It can be found/created in the settings of a new calendar. "
    "Go to Settings > Sharing > Create Link > Select Administration & All Calendars.  \n"
    "You can't change it once it's set, and it's "
    + ("already set." if camp.teamup_admin_url else "not set yet.")
)
teamup_admin_url = st.text_input(
    "Teamup admin URL",
    placeholder="https://teamup.com/ks...",
    disabled=camp.teamup_admin_url is not None,
)

# --------------------------------
st.subheader("OpenAI API keys")
st.write(
    "We use an API key for some of the notebooks in the camp. "
    "You can create one here, and don't forget to revoke it after the camp. "
)
if camp.openai_camp_service_account is None:

    def new_openai_service_account():
        edit_current_camp(openai_camp_service_account=ServiceAccount.from_name(camp.name))

    st.button("Create OpenAI API key", on_click=new_openai_service_account)
else:

    def delete_service_account():
        camp.openai_camp_service_account.delete()
        edit_current_camp(openai_camp_service_account=None)

    st.write(f"API key:\n```\n{camp.openai_camp_service_account.api_key}\n```")
    st.button("Revoke OpenAI API key", on_click=delete_service_account)


st.subheader("Participants")
st.write(
    "We need the participants names and emails here mostly to send the career planning docs. "
    "We put it as CSV to make it easier to copy-paste and edit."
)
participants = st.text_area(
    "Participants name and email CSV",
    value=camp.participants_name_and_email_csv,
    placeholder="name,email\njack,jack@ml4good.org\njulia,julia@ml4bad.com",
    help="CSV with columns 'name' and 'email'.",
    height=200,
)

if st.button("Save", type="primary") or top_save_button:
    new_data = dict(
        date=date.isoformat(),
        teamup_admin_url=teamup_admin_url,
        participants_name_and_email_csv=participants,
    )

    # Remove newdata that is None
    new_data = {k: v for k, v in new_data.items() if v not in (None, "")}
    edit_current_camp(**new_data)
