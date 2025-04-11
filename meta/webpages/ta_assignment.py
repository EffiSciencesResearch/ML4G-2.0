# %%
from collections import defaultdict
import dotenv

import streamlit as st

from utils.camp_utils import get_current_camp
from utils.streamlit_utils import State
from utils.teamup_utils import Teamup, Event, MEAL_INDICATOR

dotenv.load_dotenv()
camp = get_current_camp()
teamup = Teamup(camp)


# %%
# All of those are here to cache the data, so that we don't query at every script rerun.
# But the main logic is in the teamup class, which doesn't use any of streamlit, so it's easier to
# test and debug.
@st.cache_data()
def get_or_make_participants_key():
    return teamup.get_or_make_participants_key()


@st.cache_data()
def get_or_make_modifier_key():
    return teamup.get_or_make_modifier_key()


@st.cache_resource()  # to modify it when we edit.
def get_events():
    return teamup.get_events()


@st.cache_data()
def get_subcalendar_to_name() -> dict[int, str]:
    return teamup.get_subcalendar_to_name()


def nice_event(event: Event, day: bool = False) -> str:
    start = event.start_datetime.strftime("%H:%M")
    end = event.end_datetime.strftime("%H:%M")
    if day:
        # Tue 12 12:00 - 14:00
        start = event.start_datetime.strftime("%a %d") + " " + start

    out = f":green[{start} - {end}] {event.title}"
    if event.who:
        out += f" ({event.who})"
    return out


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
    st.write("Please go to the main page to fix this, if you use teampup.")
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


cols = st.columns(2)
with cols[0]:
    st.header("Work hours per TA")

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

with cols[1]:
    st.header("Session leads")

    session_leads = defaultdict(int)
    for event in events:
        if MEAL_INDICATOR in event.title:
            continue
        for lead in event.in_charge():
            if (
                lead in subcalendar_to_name.values()
            ):  # We don't want non-TAs, i.e. participant for meal prep
                session_leads[lead] += 1

    text = "Sessions led:"
    for lead, count in sorted(session_leads.items(), key=lambda x: -x[1]):
        text += f"\n- {lead}: {count}"

    st.write(text)


st.header("Workshops with no lead :warning:")
workshops_with_no_lead = [
    nice_event(event, day=True)
    for event in events
    if (
        not event.in_charge()
        and MEAL_INDICATOR not in event.title
        and participant_calendar_id in event.subcalendar_ids
        and not event.rrule
    )
]

if workshops_with_no_lead:
    st.write("\n- " + "\n- ".join(workshops_with_no_lead))


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
        f":green[{event.start_datetime.strftime('%H:%M')} "
        f"- {event.end_datetime.strftime('%H:%M')}] "
        f"{event.title} "
        f"({event.who})"
    )
    for col, (subcalendar_id, calendar_name) in zip(cols[1:], subcalendar_to_name.items()):
        if subcalendar_id in event.subcalendar_ids:

            if calendar_name in event.in_charge():
                # crown
                if col.button("👑", key=f"{event.id}-{subcalendar_id}-crown"):  # , type="primary"):
                    events[i] = teamup.toggle_calendar(events[i], subcalendar_id, False)
                    events[i] = teamup.toggle_in_charge(events[i], calendar_name, False)
                    st.rerun()
            else:
                if col.button("✅", key=f"{event.id}-{subcalendar_id}-ok"):
                    events[i] = teamup.toggle_in_charge(events[i], calendar_name, True)
                    st.rerun()
        else:
            if col.button("❌", key=f"{event.id}-{subcalendar_id}-no"):
                events[i] = teamup.toggle_calendar(events[i], subcalendar_id, True)
                st.rerun()
