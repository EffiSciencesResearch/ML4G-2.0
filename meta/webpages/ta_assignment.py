# %%
from collections import defaultdict
from datetime import datetime, timedelta
import dotenv

import streamlit as st

from camp_utils import get_current_camp
from streamlit_utils import State
from teamup_utils import Teamup, Event

MEAL_INDICATOR = "🥘"

dotenv.load_dotenv()
camp = get_current_camp()
teamup = Teamup(camp)


# %%
@st.cache_data()
def get_or_make_participants_key():
    return teamup.get_or_make_participants_key()


@st.cache_data()
def get_or_make_modifier_key():
    return teamup.get_or_make_modifier_key()


@st.cache_resource()
def get_events():
    today = datetime.now()
    next_year = today + timedelta(days=365)

    query = {
        "startDate": today.strftime("%Y-%m-%d"),
        "endDate": next_year.strftime("%Y-%m-%d"),
    }

    events = teamup.get("events", **query)
    return [Event.model_validate(e) for e in events["events"]]


@st.cache_data()
def get_subcalendar_to_name() -> dict[int, str]:
    subcalendars = teamup.get("subcalendars")["subcalendars"]
    return {sc["id"]: sc["name"] for sc in subcalendars}


# %%
st.set_page_config(page_title="TA Assignment", page_icon="📅", layout="wide")

state = State()

with st.sidebar:
    camp = state.auto_login()

st.markdown(
    """
# TA Assignment

This page is a tool to easily assign and remove TAs to workshops.
Toggle the ✅ and ❌ buttons to assign or remove a TA from a workshop.
"""
)

if not camp:
    st.error("Please select a camp first on the main page.")
    st.stop()

errors = camp.validate_teamup()
if errors:
    st.error(errors)
    st.write("Please go to the main page to fix this.")
    st.stop()

st.markdown(
    f"""
This directly updates [our TeamUp calendar](https://teamup.com/{get_or_make_modifier_key()}).
- Please do not share this URL outside the camp team, as it gives write access to the calendar.
- The link for the participants is https://teamup.com/{get_or_make_participants_key()}.
"""
)

with st.sidebar:
    with st.spinner("Fetching calendar data..."):
        subcalendar_to_name = get_subcalendar_to_name()
        events = get_events()

participant_calendar_id = next(
    id_ for id_, name in subcalendar_to_name.items() if name == teamup.participant_calendar_name
)

hides = []
with st.sidebar:
    if st.button("Refresh calendar"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    hide_recuring = st.checkbox("Hide recuring events", value=True)
    if st.checkbox(f"Hide meals (contain {MEAL_INDICATOR!r})", value=True, disabled=hide_recuring):
        hides.append(MEAL_INDICATOR)


st.header("Summary")

# Show:
# - number of hours assigned to each TA
# - workshops without TA

# TA to hours
ta_to_hours = defaultdict(int)
for event in events:
    if MEAL_INDICATOR in event.title:
        continue
    # if no participant
    if participant_calendar_id not in event.subcalendar_ids:
        continue
    for ta_id in event.subcalendar_ids:
        ta_to_hours[ta_id] += event.duration.total_seconds() / 3600

text = "Hours in workshops:"
for ta_id, hours in sorted(ta_to_hours.items(), key=lambda x: -x[1]):
    h, m = divmod(hours, 1)
    text += f"\n- {subcalendar_to_name[ta_id]}: {int(h)}h{int(m*60):02d}"

st.write(text)


st.header("Assignments")

col_spec = [4] + [1] * len(subcalendar_to_name)


def header():
    cols = st.columns(col_spec)
    # cols[0].write("**Event**")
    for i, subcalendar_name in enumerate(subcalendar_to_name.values()):
        cols[i + 1].write(f"**{subcalendar_name}**")


last_day = None
for i, event in enumerate(events):
    if any(h in event.title for h in hides):
        continue
    if hide_recuring and event.rrule:
        continue

    if event.start_datetime.date() != last_day:
        last_day = event.start_datetime.date()
        # Monday 9
        st.write(f"### {last_day.strftime('%A %d')}")
        header()

    cols = st.columns(col_spec)
    cols[0].write(
        f":green[{event.start_datetime.strftime('%H:%M')} - {event.end_datetime.strftime('%H:%M')}] {event.title}"
    )
    for col, subcalendar_id in zip(cols[1:], subcalendar_to_name.keys()):
        if subcalendar_id in event.subcalendar_ids:
            # emoji
            if col.button("✅", key=f"{event.id}-{subcalendar_id}-ok"):
                teamup.toggle_calendar(events, i, subcalendar_id)
                st.rerun()
        else:
            if col.button("❌", key=f"{event.id}-{subcalendar_id}-no"):
                teamup.toggle_calendar(events, i, subcalendar_id)
                st.rerun()
