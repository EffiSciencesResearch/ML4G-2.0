import os

import streamlit as st
import dotenv

from camp_utils import list_camps, Camp
from streamlit_utils import State

dotenv.load_dotenv()


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

state = State()

if state.current_camp:
    st.write(f"## Currently editing camp `{state.current_camp.name}`")
else:
    st.write("## Start by selecting a camp")


# Now two options: select a new camp or create a new one
col_select_camp, col_new_camp = st.columns(2)


with col_select_camp:
    st.header("Select a camp")
    camps = list_camps()
    camps.sort(key=lambda c: c.date, reverse=True)

    if not camps:
        st.warning("No camps found. Please create a camp first.")
    else:

        with st.container(border=True):
            camp = st.selectbox("Select camp", camps, format_func=lambda c: c.name)
            assert camp is not None

            if state.auto_login(camp.name):
                st.write("You are already logged in this camp.")
                if st.button("Log out"):
                    state.logout()
                    st.success("You were logged out.")
            else:
                password = st.text_input("Password", type="password")

                if st.button("Select this one"):
                    if state.login(camp.name, password):
                        st.success("You were logged in.")
                    else:
                        st.error("Invalid password")


with col_new_camp:
    st.header("Create a new camp")

    with st.form("create_camp"):
        name = st.text_input("Name")
        date = st.date_input("Date")

        disabled = "ENABLE_CREATE_CAMP" not in os.environ
        letsgooo = st.form_submit_button("Create camp", disabled=disabled)
        if disabled:
            st.warning("It's not yet possible to create new camp from the public portal.")

    if letsgooo:
        camp = Camp.new(name=name, date=date.strftime("%Y-%m-%d"))
        camp.save_to_disk()

        st.warning(
            "### Please write down the password for this camp. It will not be shown again.\n\n"
            f"### Password: `{camp.password}`"
        )
        st.write("You can continue configuring the camp in **Edit Camp** on the left.")

        state.login(camp.name, camp.password)
