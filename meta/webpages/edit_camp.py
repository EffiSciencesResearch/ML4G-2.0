import csv
import streamlit as st
from utils.openai_utils import ServiceAccount
from utils.camp_utils import get_current_camp, edit_current_camp
import datetime
from utils.streamlit_utils import State
from openai import OpenAI
from litellm import model_cost

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

# Password: read only
password = st.text_input("Password", disabled=True, value=camp.password)

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
    "\n\n"
    "Current **hard usage limits** are at 20$/month, but usage should generally be lower. "
    "As we are using the mostly to demo usage, and not capability, we can mostly use the cheaper ones. "
    "If more credits are needed, contact Diego or Nia."
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

    # Get available models from OpenAI API
    @st.cache_data
    def get_available_models(api_key):
        client = OpenAI(api_key=api_key)
        available_models = client.models.list()
        return list(available_models.data)

    available_models = get_available_models(camp.openai_camp_service_account.api_key)

    st.write(
        "Below are the available OpenAI models and their associated costs per 1M tokens. No other models are available, if others are needed, contact Diego."
    )

    # Create a table of available models and their costs
    table_data = []
    for model in available_models:
        if model.id in model_cost:
            costs = model_cost.get(model.id, {})
            table_data.append(
                {
                    "Model": model.id,
                    "Input Cost ($/1M tokens)": (
                        f"${costs['input_cost_per_token'] * 1000000:.2f}"
                        if "input_cost_per_token" in costs
                        else "N/A"
                    ),
                    "Output Cost ($/1M tokens)": (
                        f"${costs['output_cost_per_token'] * 1000000:.2f}"
                        if "output_cost_per_token" in costs
                        else "N/A"
                    ),
                    "Max Tokens": costs.get("max_tokens", "N/A"),
                }
            )

    table_data.sort(key=lambda x: x["Input Cost ($/1M tokens)"])
    st.table(table_data)

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
        teamup_admin_url=teamup_admin_url,
        participants_name_and_email_csv=participants,
        feedback_sheet_url=feedback_sheet_url,
    )

    # Remove newdata that is None
    new_data = {k: v for k, v in new_data.items() if v not in (None, "")}
    edit_current_camp(**new_data)
