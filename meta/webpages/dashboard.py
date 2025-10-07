import streamlit as st
import dotenv
from streamlit_product_card import product_card

from utils.streamlit_utils import State

dotenv.load_dotenv()


state = State()

with st.sidebar:
    state.login_form(key="sidebar_form")


name = "" if state.current_camp is None else f" *{state.current_camp.name}*"
st.title(f"Welcome to the ML4G{name} tools portal!")

st.markdown(
    """This website is and will grow into a collection of handy tools to
help run the ML4G bootcamps.

We aim to make those tools self-explanatory, so that they
require *no* external documentation, but if you find
yourself confused or things are broken, please reach out
to Diego or the ML4G team on Slack.

Enjoy! :rocket:
"""
)

st.subheader("Most useful tools")
cols = st.columns(3)
tools = [
    (
        "webpages/edit_camp.py",
        "🏠",
        "Get OpenAI keys, set participants and variables for everyone.",
    ),
    ("webpages/career_docs.py", "📄", "Auto duplicate google docs for each participant."),
    ("webpages/one_on_one_scheduler.py", "👥", "Schedule one-on-ones with participants."),
]

for i, (page, icon, description) in enumerate(tools):
    with cols[i]:
        title = page.split("/")[-1].replace("_", " ").replace(".py", "").title()
        was_clicked = product_card(
            product_name=f"{icon} {title}",
            description=description,
            key=f"tool_{i}",
        )
        if was_clicked:
            st.switch_page(page)


if not state.current_camp:
    st.write("## 👈🏻 Start by selecting a camp on the left sidebar")
    st.write(
        "On a small screen the sidebar might be collapsed, but you'll find a `>` button to expand it on the top right. "
        "If you're an organiser/teacher/TA, a password should have been given to you by the main camp organizer, if not, contact them."
    )
