from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field


# --- Discriminated Union for Questions ---


class BaseQuestionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    description: str | None = None
    mandatory: bool = False


class TextQuestionConfig(BaseQuestionConfig):
    kind: Literal["text"]


class ParagraphQuestionConfig(BaseQuestionConfig):
    kind: Literal["paragraph"]


class ScaleQuestionConfig(BaseQuestionConfig):
    kind: Literal["scale"]
    low: int = 1
    high: int = 10
    low_label: str | None = None
    high_label: str | None = None


class ChoiceQuestionConfig(BaseQuestionConfig):
    kind: Literal["choice"]
    choices: list[str] | None = None
    dropdown: bool = False


AnyQuestionConfig = Union[
    TextQuestionConfig,
    ParagraphQuestionConfig,
    ScaleQuestionConfig,
    ChoiceQuestionConfig,
]


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    reading_group: bool = False


class DayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meme: str | None = None
    sessions: list[SessionConfig]
    day_questions: list[AnyQuestionConfig] = Field(default_factory=list)


class CampConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camp_name: str
    drive_folder_id: str = ""
    teachers: list[str] = Field(default_factory=list)
    form_description: str = ""
    pre_questions: list[AnyQuestionConfig] = Field(default_factory=list)
    timetable: dict[str, DayConfig]
    post_questions: list[AnyQuestionConfig] = Field(default_factory=list)


# If you modify the models, you can regenerate the schema by running:
# uv run python models.py
if __name__ == "__main__":
    import json
    from pathlib import Path

    def main():
        """Generate JSON schema from CampConfig model."""
        schema = CampConfig.model_json_schema()
        schema_path = Path(__file__).parent / "config.schema.json"
        schema_path.write_text(json.dumps(schema, indent=2) + "\n")
        print(f"✅ Schema generated at {schema_path}")

    main()
