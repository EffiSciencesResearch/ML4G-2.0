import os
import streamlit as st
from dotenv import load_dotenv

from utils.streamlit_utils import State
from utils.camp_utils import Camp

load_dotenv()


state = State()


st.title("Create a new camp")

with st.form("create_camp"):
    name = st.text_input("Name")
    date = st.date_input("Date")
    st.write("The admin password can be found at the end of the TA & Teacher guide.")
    admin_password = st.text_input("Admin password to create the camp", type="password")

    letsgooo = st.form_submit_button("Create camp")

if letsgooo:
    if admin_password is not None and admin_password == os.getenv("ML4G_PORTAL_ADMIN_PASSWORD"):
        camp = Camp.new(name=name, date=date.strftime("%Y-%m-%d"))
        camp.save_to_disk()

        st.warning(
            "### Please write down the password for this camp. It will not be shown again if you log out.\n\n"
            f"### Password: `{camp.password}`"
        )
        st.write("You can continue configuring the camp in **Edit Camp** on the left.")

        state.login(camp.name, camp.password)

    else:
        st.error("Invalid admin password")
