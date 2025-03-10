# %%
import os
from pydantic import BaseModel
import requests
import json
from datetime import datetime, timedelta
import dotenv

from utils.camp_utils import Camp

dotenv.load_dotenv()

PARTICIPANTS_CALENDAR_NAME = "Participants"
BASE_API_URL = "https://api.teamup.com"
MEAL_INDICATOR = "🥘"


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

    def in_charge(self):
        return [name.strip() for name in self.who.split("&") if name.strip()]


class Teamup:
    def __init__(
        self,
        camp: Camp,
        participant_calendar_name: str = PARTICIPANTS_CALENDAR_NAME,
        base_api_url: str = BASE_API_URL,
    ):
        self.camp = camp
        self.participant_calendar_name = participant_calendar_name
        self.base_api_url = base_api_url

    def get(self, *path: str, **query: str):
        url = "/".join([self.base_api_url, self.camp.teamup_admin_calendar_key, *path])
        headers = {
            "Teamup-Token": os.environ["TEAMUP_API_KEY"],
        }
        print("GET", url, query)
        response = requests.get(url, headers=headers, params=query)
        if response.status_code != 200:
            print(response.json())
            raise Exception("Failed to get 200 from Teamup")
        return response.json()

    def put(self, *path: str, data: dict, post: bool = False):
        url = "/".join([self.base_api_url, self.camp.teamup_admin_calendar_key, *path])
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

    def get_events(self) -> list[Event]:
        start = self.camp.start_datetime - timedelta(days=10)
        end = self.camp.start_datetime + timedelta(days=20)

        query = {
            "startDate": start.strftime("%Y-%m-%d"),
            "endDate": end.strftime("%Y-%m-%d"),
        }

        events = self.get("events", **query)
        return [Event.model_validate(e) for e in events["events"]]

    def edit_event(self, event: Event) -> Event:
        response = self.put("events", event.id, data=event.model_dump())
        return Event.model_validate(response["event"])

    def toggle_calendar(self, event: Event, calendar_id: int, value: bool) -> Event:
        ids = set(event.subcalendar_ids)
        if value:
            ids.add(calendar_id)
        else:
            ids.discard(calendar_id)
        event.subcalendar_ids = list(ids)

        return self.edit_event(event)

    def toggle_in_charge(self, event: Event, name: str, value: bool) -> Event:
        who = set(event.in_charge())
        if value:
            who.add(name)
        else:
            who.discard(name)

        event.who = "&".join(sorted(who))
        return self.edit_event(event)

    def get_or_make_key(self, desired_data: dict, name: str) -> str:
        keys = self.get("keys")

        # Try to find a key which has desired_data
        for key in keys["keys"]:
            if all(key[k] == v for k, v in desired_data.items()):
                return key["key"]

        # If we didn't find a key, create one
        desired_data["name"] = name
        response = self.put("keys", data=desired_data, post=True)
        return response["key"]["key"]

    def get_or_make_modifier_key(self) -> str:
        # Try to find a key which has:
        desired_data = {
            "active": True,
            "admin": False,
            "role": "modify",
            "share_type": "all_subcalendars",
        }

        return self.get_or_make_key(desired_data, "Write access for organizing team")

    def get_or_make_participants_key(self) -> str:
        subcalendars = self.get("subcalendars")["subcalendars"]
        participant_subcalendar_id = next(
            sc["id"] for sc in subcalendars if sc["name"] == self.participant_calendar_name
        )

        desired_data = {
            "active": True,
            "admin": False,
            "share_type": "selected_subcalendars",
            "subcalendar_permissions": {str(participant_subcalendar_id): "read_only"},
        }

        return self.get_or_make_key(desired_data, "Read access for participants")

    def get_subcalendar_to_name(self) -> dict[int, str]:
        subcalendars = self.get("subcalendars")["subcalendars"]
        return {sc["id"]: sc["name"] for sc in subcalendars}
