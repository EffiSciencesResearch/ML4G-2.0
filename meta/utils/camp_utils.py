import csv
import datetime
from pathlib import Path
import string
from pydantic import BaseModel
import random

from utils.openai_utils import ServiceAccount

CAMPS_DIR = Path(__file__).parent.parent / "camps"


class Camp(BaseModel):
    name: str
    password: str
    date: str
    participants_name_and_email_csv: str = "name,email\n"
    openai_camp_service_account: ServiceAccount | None = None
    feedback_sheet_url: str | None = None
    # If you add a new field here, remember to add it to edit_camp.py too.

    @classmethod
    def new(cls, name: str, date: str) -> "Camp":
        password = "".join(random.choices(string.ascii_letters, k=16))
        return cls(name=name, password=password, date=date)

    @property
    def start_datetime(self):
        return datetime.datetime.fromisoformat(self.date)

    def participants_list(self) -> list[str]:
        reader = csv.DictReader(self.participants_name_and_email_csv.splitlines())
        return list(row["name"] for row in reader)
