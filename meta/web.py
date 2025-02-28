import datetime
import streamlit as st
from camp_utils import list_camps, Camp, CAMPS_DIR, get_current_camp, set_current_camp


st.title("Welcome to the ML4G tools portal!")

st.markdown(
    """
This page is and will grow into a collection of handy tools to
help run the ML4G bootcamps.

We aim to make those tools self-explanatory, so that they
require *no* external documentation, but if you find
yourself confused or things are broken, please reach out
to Diego or the ML4G team on Slack.

This page lets you select which camp to use for all the tools and create a new one.

Enjoy! :rocket:
"""
)


if get_current_camp():
    st.write(f"## Currently editing camp `{get_current_camp().name}`")
else:
    st.write("## Start by selecting a camp")


# Now two options: select a new camp or create a new one
col_select_camp, col_new_camp = st.columns(2)


with col_select_camp:
    st.header("Select a camp")
    camps = list_camps()

    if not camps:
        st.warning("No camps found. Please create a camp first.")
        st.stop()

    with st.form("select_camp"):
        sorted_camps = sorted(camps, key=lambda c: camps[c].date, reverse=True)
        camp_file = st.selectbox("Select camp", sorted_camps, format_func=lambda c: camps[c].name)
        camp = camps[camp_file]
        password = st.text_input("Password", type="password")

        if st.form_submit_button("Select this one"):
            if password != camp.password:
                st.error("Invalid password")
            else:
                set_current_camp(camp, camp_file)
                st.rerun()


with col_new_camp:
    st.header("Create a new camp")

    with st.form("create_camp"):
        name = st.text_input("Name")
        date = st.date_input("Date")
        letsgooo = st.form_submit_button("Create camp")

    if letsgooo:
        camp = Camp.new(name=name, date=date.strftime("%Y-%m-%d"))
        now = datetime.datetime.now()
        camp_file = CAMPS_DIR / f"{now.strftime('%Y-%m-%d %H:%M:%S')} {name}.json"

        st.warning(
            "### Please write down the password for this camp. It will not be shown again.\n\n"
            f"### Password: `{camp.password}`"
        )

        camp_file.write_text(camp.model_dump_json())
        camps[camp_file] = camp

        set_current_camp(camp, camp_file)
