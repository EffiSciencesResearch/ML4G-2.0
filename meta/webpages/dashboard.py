from datetime import datetime, timezone

import streamlit as st
import dotenv
from streamlit_pills import pills as st_pills

from utils.streamlit_utils import State
from utils.teamup_utils import Teamup, MEAL_INDICATOR

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


if not state.current_camp:
    st.write("## 👈🏻 Start by selecting a camp on the left sidebar")
    st.write(
        "On a small screen the sidebar might be collapsed, but you'll find a `>` button to expand it on the top right. "
        "If you're an organiser/teacher/TA, a password should have been given to you by the main camp organizer, if not, contact them."
    )
    st.stop()


st.subheader("Most useful tools")
cols = st.columns(3)
cols[0].page_link("webpages/edit_camp.py", use_container_width=True, icon="🏠")
cols[1].page_link("webpages/career_docs.py", use_container_width=True, icon="📄")
cols[2].page_link("webpages/one_on_one_scheduler.py", use_container_width=True, icon="👥")


teamup = Teamup(state.current_camp)

if state.current_camp.validate_teamup() is not None:
    st.divider()
    st.warning(
        "The dashboard works only when using teamup, and it shows the next events for each TA. You can add a teamup URL in the Edit Camp page, if you do use teamup."
    )
    st.stop()


@st.cache_data()
def get_events():
    return teamup.get_events()


@st.cache_data()
def get_subcalendar_to_name():
    return teamup.get_subcalendar_to_name()


events = get_events()

st.sidebar.button("Refresh calendar", on_click=get_events.clear)

subcalendar_to_name = get_subcalendar_to_name()

try:
    default_index = list(subcalendar_to_name.values()).index(
        state.per_browser_settings.dashboard_name
    )
except ValueError:
    default_index = 0

current_user = st_pills(
    "View dashboard as",
    list(subcalendar_to_name),
    format_func=lambda x: subcalendar_to_name[x],
    index=default_index,
    key="dashboard_name",
)
current_user_name = subcalendar_to_name[current_user]
state.save_settings(dashboard_name=current_user_name)

hide_meals = st.sidebar.checkbox(f"Hide meals ({MEAL_INDICATOR})", value=True)

# Columns:
# 1. List of next events for this person
# 2. List of next events this person is in charge of

col_next_events, col_in_charge = st.columns(2)

next_events = []
in_charge_events = []
now = datetime.now(timezone.utc)

for event in events:
    # Event is in the past
    if event.end_datetime < now:
        continue
    if hide_meals and MEAL_INDICATOR in event.title:
        continue

    if current_user in event.subcalendar_ids:
        next_events.append(event)
    if current_user_name in event.in_charge():
        in_charge_events.append(event)


def show_event_list(events):
    if not events:
        st.write("No events")
        return

    text = ""
    for event in events:
        time_until = event.start_datetime - now
        # Show as days? hh:mm
        if time_until.days:
            time_until_str = f"{time_until.days} day{'s' if time_until.days > 1 else ''} {time_until.seconds // 3600}h"
        else:
            time_until_str = f"{time_until.seconds // 3600}h {time_until.seconds % 3600 // 60}m"

        # if in < 24h, title in red
        if time_until.days == 0:
            title = f":primary[{event.title}]"
        else:
            title = event.title

        text += f"- **{title}** :grey[in {time_until_str}]\n"

    st.write(text)


with col_next_events:
    st.header(f"{current_user_name}'s next events")

    show_event_list(next_events)

with col_in_charge:
    st.header(f"{current_user_name} is in charge of")

    show_event_list(in_charge_events)
