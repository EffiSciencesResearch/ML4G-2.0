# %%
import os
import pprint
from openai import BaseModel
import requests
import dotenv

dotenv.load_dotenv()

project_id = os.getenv("OPENAI_CAMPS_PROJECT_ID")
admin_key = os.getenv("OPENAI_ADMIN_KEY")


class ServiceAccount(BaseModel):
    id: str
    api_key: str

    @classmethod
    def from_name(cls, name: str) -> "ServiceAccount":
        # https://platform.openai.com/docs/api-reference/project-service-accounts/create

        assert (
            project_id
        ), "Please set OPENAI_CAMPS_PROJECT_ID to the project in which to create the account"
        assert admin_key, "Please set OPENAI_ADMIN_KEY with an administrator (not API!) key"

        response = requests.post(
            f"https://api.openai.com/v1/organization/projects/{project_id}/service_accounts",
            headers={
                "Authorization": f"Bearer {admin_key}",
                "Content-Type": "application/json",
            },
            json={"name": name},
        )
        response.raise_for_status()
        data = response.json()

        return cls(id=data["id"], api_key=data["api_key"]["value"])

    def delete(self):
        # Delete https://api.openai.com/v1/organization/service_accounts/{account_id}
        assert admin_key, "Please set OPENAI_ADMIN_KEY with an administrator (not API!) key"

        response = requests.delete(
            f"https://api.openai.com/v1/organization/service_accounts/{self.id}",
            headers={
                "Authorization": f"Bearer {admin_key}",
            },
        )
        response.raise_for_status()


def list_projects():
    # Get https://api.openai.com/v1/organization/projects
    assert admin_key, "Please set OPENAI_ADMIN_KEY with an administrator (not API!) key"

    response = requests.get(
        "https://api.openai.com/v1/organization/projects",
        headers={
            "Authorization": f"Bearer {admin_key}",
        },
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    r = list_projects()
    pprint(r)
