import streamlit as st
from camp_utils import get_current_campfile, get_current_camp, set_current_camp
import datetime


camp = get_current_camp()
campfile = get_current_campfile()
if not camp:
    st.error("Please select a camp first on the main page.")
    st.stop()

st.write(f"## Currently editing camp `{camp.name}`")

with st.form("camp-edit", border=False):
    name = st.text_input("Name", value=camp.name)
    date = st.date_input("Date", value=datetime.datetime.fromisoformat(camp.date))

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

    st.write(
        "We need the participants names and emails here mostly to send the career planning docs. "
        "We put it as CSV to make it easier to copy-paste and edit."
    )
    participants = st.text_area(
        "Participants name and email CSV",
        value=camp.participants_name_and_email_csv or "name,email\n",
        help="CSV with columns 'name' and 'email'.",
        height=200,
    )

    if st.form_submit_button("Save"):
        new_data = dict(
            name=name,
            date=date.isoformat(),
            teamup_admin_url=teamup_admin_url,
            participants_name_and_email_csv=participants,
        )

        # Remove newdata that is None
        new_data = {k: v for k, v in new_data.items() if v not in (None, "")}
        st.write(new_data)

        print("Old camp", camp)
        new_camp = camp.model_copy(update=new_data)
        print("New camp", new_camp)

        campfile.write_text(new_camp.model_dump_json())
        set_current_camp(new_camp, campfile)
