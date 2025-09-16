from typing import Literal, Union

from pydantic import BaseModel, Field


# --- Discriminated Union for Questions ---


class BaseQuestionConfig(BaseModel):
    text: str
    mandatory: bool = False


class TextQuestionConfig(BaseQuestionConfig):
    kind: Literal["text"] = "text"


class ParagraphQuestionConfig(BaseQuestionConfig):
    kind: Literal["paragraph"] = "paragraph"


class ScaleQuestionConfig(BaseQuestionConfig):
    kind: Literal["scale"] = "scale"
    low: int = 1
    high: int = 10
    low_label: str | None = None
    high_label: str | None = None


class ChoiceQuestionConfig(BaseQuestionConfig):
    kind: Literal["choice"] = "choice"
    choices: list[str] | None = None
    dropdown: bool = False


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


# If you modify the models, you can regenerate the schema by running:
# uv run python models.py
if __name__ == "__main__":
    import json
    from pathlib import Path

    def main():
        """Generate JSON schema from CampConfig model."""
        schema = CampConfig.model_json_schema()
        schema_path = Path(__file__).parent / "config.schema.json"
        schema_path.write_text(json.dumps(schema, indent=2))
        print(f"✅ Schema generated at {schema_path}")

    main()
