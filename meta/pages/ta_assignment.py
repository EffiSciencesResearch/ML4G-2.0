# %%
from collections import defaultdict
import os
from pydantic import BaseModel
import requests
import json
from datetime import datetime, timedelta
import dotenv

import streamlit as st

from camp_utils import get_current_camp


dotenv.load_dotenv()
camp = get_current_camp()


PARTICIPANTS_CALENDAR_NAME = "Participants"
MEAL_INDICATOR = "🥘"
BASE_API_URL = "https://api.teamup.com"


class Event(BaseModel):
    id: str
    series_id: int | None
    remote_id: int | None
    subcalendar_id: int
    subcalendar_ids: list[int]
    all_day: bool
    rrule: str | None
    title: str
    who: str
    location: str
    notes: str
    version: str
    readonly: bool
    tz: str
    attachments: list[str]
    start_dt: str
    end_dt: str
    ristart_dt: str | None
    rsstart_dt: str | None
    creation_dt: str
    update_dt: str | None
    delete_dt: str | None
    signup_enabled: bool
    comments_enabled: bool
    comments_visibility: str
    comments: list

    @property
    def start_datetime(self):
        return datetime.fromisoformat(self.start_dt)

    @property
    def end_datetime(self):
        return datetime.fromisoformat(self.end_dt)

    @property
    def duration(self):
        return self.end_datetime - self.start_datetime


def api_get(*path: str, **query: str):
    url = "/".join([BASE_API_URL, camp.teamup_admin_calendar_key, *path])
    headers = {
        "Teamup-Token": os.environ["TEAMUP_API_KEY"],
    }
    print("GET", url, query)
    response = requests.get(url, headers=headers, params=query)
    if response.status_code != 200:
        print(response.json())
        raise Exception("Failed to get 200 from Teamup")
    return response.json()


def api_put(*path: str, data: dict, post: bool = False):
    url = "/".join([BASE_API_URL, camp.teamup_admin_calendar_key, *path])
    headers = {
        "Teamup-Token": os.environ["TEAMUP_API_KEY"],
        "Content-Type": "application/json",
    }
    if post:
        response = requests.post(url, headers=headers, data=json.dumps(data))
    else:
        response = requests.put(url, headers=headers, data=json.dumps(data))
    if not response.ok:
        print(response.json())
        raise Exception(f"Got {response.status_code} from Teamup")
    return response.json()


def toggle_calendar(events: list[Event], event_idx: int, calendar_id: int):
    event = events[event_idx]
    if calendar_id in event.subcalendar_ids:
        event.subcalendar_ids.remove(calendar_id)
    else:
        event.subcalendar_ids.append(calendar_id)
    response = api_put("events", event.id, data=event.model_dump())

    # Update the event in the list
    events[event_idx] = Event.model_validate(response["event"])


def get_or_make_key(desired_data: dict, name: str) -> str:
    keys = api_get("keys")

    # Try to find a key which has desired_data
    for key in keys["keys"]:
        if all(key[k] == v for k, v in desired_data.items()):
            return key["key"]

    # If we didn't find a key, create one
    desired_data["name"] = name
    response = api_put("keys", data=desired_data, post=True)
    return response["key"]["key"]


# %%


@st.cache_data()
def get_or_make_modifier_key() -> str:
    # Try to find a key which has:
    desired_data = {
        "active": True,
        "admin": False,
        "role": "modify",
        "share_type": "all_subcalendars",
    }

    return get_or_make_key(desired_data, "Write access for organizing team")


@st.cache_data()
def get_or_make_participants_key() -> str:
    subcalendars = api_get("subcalendars")["subcalendars"]
    participant_subcalendar_id = next(
        sc["id"] for sc in subcalendars if sc["name"] == PARTICIPANTS_CALENDAR_NAME
    )

    desired_data = {
        "active": True,
        "admin": False,
        "share_type": "selected_subcalendars",
        "subcalendar_permissions": {str(participant_subcalendar_id): "read_only"},
    }

    return get_or_make_key(desired_data, "Read access for participants")


@st.cache_resource()
def get_events():
    today = datetime.now()
    next_year = today + timedelta(days=365)

    query = {
        "startDate": today.strftime("%Y-%m-%d"),
        "endDate": next_year.strftime("%Y-%m-%d"),
    }

    events = api_get("events", **query)
    return [Event.model_validate(e) for e in events["events"]]


@st.cache_data()
def get_subcalendar_to_name() -> dict[int, str]:
    subcalendars = api_get("subcalendars")["subcalendars"]
    return {sc["id"]: sc["name"] for sc in subcalendars}


# %%
st.set_page_config(page_title="TA Assignment", page_icon="📅", layout="wide")
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
    id_ for id_, name in subcalendar_to_name.items() if name == PARTICIPANTS_CALENDAR_NAME
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
                toggle_calendar(events, i, subcalendar_id)
                st.rerun()
        else:
            if col.button("❌", key=f"{event.id}-{subcalendar_id}-no"):
                toggle_calendar(events, i, subcalendar_id)
                st.rerun()
