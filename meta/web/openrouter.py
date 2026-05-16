import datetime
import os

import dotenv
import requests
from pydantic import BaseModel

dotenv.load_dotenv()

provisioning_key = os.getenv("OPENROUTER_PROVISIONING_KEY")

CAMP_KEY_LIMIT_USD = 20.0
CAMP_KEY_DURATION_DAYS = 14


class OpenRouterAPIKey(BaseModel):
    hash: str
    api_key: str
    name: str
    limit: float
    expires_at: datetime.datetime | None = None

    @classmethod
    def from_name(cls, name: str, limit: float = CAMP_KEY_LIMIT_USD) -> "OpenRouterAPIKey":
        # https://openrouter.ai/docs/features/provisioning-api-keys
        assert provisioning_key, (
            "Please set OPENROUTER_PROVISIONING_KEY with a provisioning key from "
            "https://openrouter.ai/settings/provisioning-keys"
        )

        expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            days=CAMP_KEY_DURATION_DAYS
        )
        expires_at_str = expires_at.strftime("%Y-%m-%dT%H:%M:%SZ")

        response = requests.post(
            "https://openrouter.ai/api/v1/keys",
            headers={
                "Authorization": f"Bearer {provisioning_key}",
                "Content-Type": "application/json",
            },
            json={"name": name, "limit": limit, "expires_at": expires_at_str},
        )
        response.raise_for_status()
        data = response.json()
        key_data = data["data"]
        return cls(
            hash=key_data["hash"],
            api_key=data["key"],
            name=key_data["name"],
            limit=key_data.get("limit") or limit,
            expires_at=expires_at,
        )

    def delete(self):
        assert provisioning_key, "Please set OPENROUTER_PROVISIONING_KEY"
        response = requests.delete(
            f"https://openrouter.ai/api/v1/keys/{self.hash}",
            headers={"Authorization": f"Bearer {provisioning_key}"},
        )
        response.raise_for_status()

    def fetch_usage(self) -> dict:
        assert provisioning_key, "Please set OPENROUTER_PROVISIONING_KEY"
        response = requests.get(
            f"https://openrouter.ai/api/v1/keys/{self.hash}",
            headers={"Authorization": f"Bearer {provisioning_key}"},
        )
        response.raise_for_status()
        return response.json()["data"]
