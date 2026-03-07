# %%
import os
from abc import ABC, abstractmethod
from pathlib import Path

import boto3
from dotenv import load_dotenv

from utils.camp_utils import Camp

load_dotenv()

CAMP_DIR = Path(__file__).parent.parent / "camps"


class Storage(ABC):
    @abstractmethod
    def save_camp(self, camp: Camp):
        pass

    @abstractmethod
    def load_camp(self, name: str) -> Camp:
        pass

    @abstractmethod
    def list_camps(self) -> list[str]:
        pass


class LocalStorage(Storage):
    def __init__(self, base_dir: Path = CAMP_DIR):
        self.base_dir = base_dir
        print(f"Using local storage: {self.base_dir}")

    def save_camp(self, camp: Camp):
        campfile = self.base_dir / f"{camp.name}.json"
        campfile.write_text(camp.model_dump_json())

    def load_camp(self, name: str) -> Camp:
        campfile = self.base_dir / f"{name}.json"
        return Camp.model_validate_json(campfile.read_text())

    def list_camps(self) -> list[str]:
        return [camp.stem for camp in self.base_dir.glob("*.json") if camp.is_file()]


class S3Storage(Storage):
    PREFIX = "camps/"

    def __init__(self):
        self.s3 = boto3.resource("s3")
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME is not set")
        self.bucket = self.s3.Bucket(self.bucket_name)

        print(f"Using S3 bucket: {self.bucket_name}")

    def save_camp(self, camp: Camp):
        key = self.name_to_key(camp.name)
        self.bucket.put_object(
            Key=key,
            Body=camp.model_dump_json().encode("utf-8"),
            ContentType="application/json",
        )

    def load_camp(self, name: str) -> Camp:
        key = self.name_to_key(name)
        obj = self.bucket.Object(key)
        obj_data = obj.get()["Body"].read().decode("utf-8")
        return Camp.model_validate_json(obj_data)

    def list_camps(self) -> list[str]:
        return [self.key_to_name(obj.key) for obj in self.bucket.objects.filter(Prefix=self.PREFIX)]

    def name_to_key(self, name: str) -> str:
        return self.PREFIX + name + ".json"

    def key_to_name(self, key: str) -> str:
        return key.removeprefix(self.PREFIX).removesuffix(".json")


def get_storage() -> Storage:
    if os.getenv("S3_BUCKET_NAME"):
        return S3Storage()
    return LocalStorage()


def upload_local():
    local = LocalStorage()
    s3 = S3Storage()
    for camp in local.list_camps():
        s3.save_camp(local.load_camp(camp))

    print("Camps in S3:")
    print(s3.list_camps())
