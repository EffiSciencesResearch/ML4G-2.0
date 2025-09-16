from typing import Literal, Union

from pydantic import BaseModel, Field


# --- Discriminated Union for Questions ---


class BaseQuestionConfig(BaseModel):
    text: str
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


AnyQuestionConfig = Union[
    TextQuestionConfig,
    ParagraphQuestionConfig,
    ScaleQuestionConfig,
    ChoiceQuestionConfig,
]


class SessionConfig(BaseModel):
    name: str
    reading_group: bool = False


class DayConfig(BaseModel):
    meme: str | None = None
    sessions: list[SessionConfig]
    day_questions: list[AnyQuestionConfig] = Field(default_factory=list)


class CampConfig(BaseModel):
    camp_name: str
    drive_folder_id: str = ""
    teachers: list[str] = Field(default_factory=list)
    form_description: str = ""
    pre_questions: list[AnyQuestionConfig] = Field(default_factory=list)
    timetable: dict[str, DayConfig]
    post_questions: list[AnyQuestionConfig] = Field(default_factory=list)


if __name__ == "__main__":
    import json
    from pathlib import Path

    def main():
        """Generate JSON schema from CampConfig model."""
        schema = CampConfig.model_json_schema()
        # a bit weird to have the schema inside the folder, but whatever
        schema_path = Path(__file__).parent / "config.schema.json"
        schema_path.write_text(json.dumps(schema, indent=2))
        print(f"✅ Schema generated at {schema_path}")

    main()
