import csv
from pathlib import Path
import streamlit as st
from utils.camp_utils import get_current_camp
from utils.google_utils import extract_id_from_url, SimpleGoogleAPI


camp = get_current_camp()

SERVICE_ACCOUNT_FILE = Path(__file__).parent.parent / "service_account_token.json"
API = SimpleGoogleAPI(SERVICE_ACCOUNT_FILE)

st.write(
    """
# Career planning docs

This page help to duplicate the career planning docs for all participants
and share them with each of them individually.

First you can edit the participants list below right here, in CSV format if needed.

The format expected is:
```
name,email
Sam,sam@openai.com
Demis,demis@deepmind.google
```
"""
)

participants = st.text_area(
    "Participants name and email CSV - temporary override",
    value=camp.participants_name_and_email_csv,
    placeholder="name,email\njack,jack@ml4good.org\njulia,julia@ml4bad.com",
    help="CSV with columns 'name' and 'email'.",
    height=200,
)
if participants != camp.participants_name_and_email_csv:
    st.rerun()

template_url = st.text_input(
    "Google Docs template URL",
)
folder_url = st.text_input(
    "Folder to put the 1-1 docs",
)


if not template_url or not folder_url:
    st.stop()


@st.cache_data()
def get_doc_name(url):
    return API.get_file_name(extract_id_from_url(url))


doc_name = get_doc_name(template_url)
to_replace = "[NAME]"


email_to_name = {}
reader = csv.DictReader(participants.splitlines())
assert set(list(reader.fieldnames)) == {
    "email",
    "name",
}, "Invalid column names. Must be 'email' and 'name'."
for row in reader:
    email_to_name[row["email"]] = row["name"]

st.write(
    "### Checklist before duplicating\n"
    "You need to make sure everything is correct, as it will send emails to each participants and mistakes are costly."
)


ok = st.checkbox(f"I checked that the {len(email_to_name)} names and emails are correct")
st.table([(email, name) for email, name in email_to_name.items()])
ok &= st.checkbox(f"The template is the correct one: {template_url}")
ok &= st.checkbox(f"The folder for 1-1 docs is correct: {folder_url}")
ok &= st.checkbox(
    "The folder for 1-1 is not accessible to participants, nor to people outside the camp team."
)
ok &= st.checkbox(
    f"The filename `{doc_name}` contains `{to_replace}`.",
    value=to_replace in doc_name,
    disabled=True,
)
e = st.checkbox(
    f"The template contains the placeholder `{to_replace}`, precisely and not something else."
)

if st.button(
    "Duplicate template and share with participants",
    type="primary",
    disabled=not ok,
    use_container_width=True,
):
    for email, name in email_to_name.items():
        st.write(f"Processing document for {name} ({email})", end="... ", flush=True)
        new_name = doc_name.replace("[NAME]", name)
        copied_doc_id = API.copy_file(
            extract_id_from_url(template_url), extract_id_from_url(folder_url), new_name
        )
        API.replace_in_document(copied_doc_id, "[NAME]", name)
        API.share_document(copied_doc_id, email)
        st.write("✅")
