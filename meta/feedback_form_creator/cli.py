#!/usr/bin/env python3
"""
Main script to create daily feedback forms for a multi-day camp.
"""

import sys
import os
from typing import Union

from meta.feedback_form_creator.forms_utils import (
    get_forms_service,
    get_drive_service,
    load_config,
    create_base_form,
    add_questions_to_form,
    create_question_from_config,
    upload_image_to_drive_and_get_url,
    create_image_item_with_url,
    move_file_to_folder,
)
from meta.feedback_form_creator.models import (
    AnyQuestionConfig,
    CampConfig,
    ChoiceQuestionConfig,
    DayConfig,
    ImageItem,
    ParagraphQuestionConfig,
    ScaleQuestionConfig,
    TextQuestionConfig,
)


PlanItem = Union[AnyQuestionConfig, ImageItem]


def build_question_plan(config: CampConfig, day_config: DayConfig) -> list[PlanItem]:
    """Build the ordered list of form items for a single day.

    Single source of truth for what the form will contain. Used by both the
    creator (which then calls the Google Forms API for each item) and the
    web-page preview (which renders titles).
    """
    plan: list[PlanItem] = list(config.pre_questions)

    for session in day_config.sessions:
        plan.append(
            ScaleQuestionConfig(
                kind="scale",
                text=f"How would you rate the '{session.name}' session?",
                description=session.description,
                low=1,
                high=5,
                low_label="Poor",
                high_label="Excellent",
            )
        )
        if session.reading_group:
            plan.append(
                ChoiceQuestionConfig(
                    kind="choice",
                    text=f"Which teacher facilitated the '{session.name}' reading group?",
                    choices=config.teachers,
                    mandatory=False,
                )
            )
        plan.append(
            ParagraphQuestionConfig(
                kind="paragraph",
                text=f"Any additional feedback on '{session.name}'?",
                mandatory=False,
            )
        )

    plan.extend(day_config.day_questions)
    plan.extend(config.post_questions)

    if day_config.meme:
        plan.append(ImageItem(filename=day_config.meme))
        plan.append(
            TextQuestionConfig(
                kind="text",
                text="Provide a caption for this meme that describes your day!",
                mandatory=False,
            )
        )

    return plan


def create_daily_feedback_form(
    service,
    drive_service,
    config: CampConfig,
    day_name: str,
    day_number: int,
    day_config: DayConfig,
):
    """Create a feedback form for a specific day."""
    form_title = f"{config.camp_name} Day {day_number} Feedback Form"
    print(f"Creating form: {form_title}")

    form_id, form = create_base_form(service, form_title, config.form_description)

    if config.drive_folder_id:
        move_file_to_folder(drive_service, form_id, config.drive_folder_id)
        print("  ✓ Form moved to Drive folder")

    memes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memes")

    items = []
    for plan_item in build_question_plan(config, day_config):
        if isinstance(plan_item, ImageItem):
            print(f"  Adding image: {plan_item.filename}")
            url = upload_image_to_drive_and_get_url(
                drive_service, os.path.join(memes_dir, plan_item.filename)
            )
            print(f"    Uploaded to Drive: {url}")
            items.append(create_image_item_with_url(url))
        else:
            items.append(create_question_from_config(plan_item))

    add_questions_to_form(service, form_id, items)

    print(f"  ✓ Form created with {len(items)} items.")
    print(f"  Edit URL: https://docs.google.com/forms/d/{form_id}/edit")
    print(f"  Live URL: {form['responderUri']}")
    print()

    return form_id, form


def main():
    """Main function to create feedback form for a specific day."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")

    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found!")
        sys.exit(1)

    config = load_config(config_path)

    # Ask user which day to create
    timetable = config.timetable
    print("Available days:")
    for i, day_name in enumerate(timetable.keys(), 1):
        print(f"  {i}: {day_name}")
    print()

    while True:
        try:
            day_input = input("Which day should be created? Enter the number: ").strip()
            day_number = int(day_input)
            if 1 <= day_number <= len(timetable):
                break
            else:
                print(f"Please enter a number between 1 and {len(timetable)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

    # Get the selected day
    day_name = list(timetable.keys())[day_number - 1]
    day_config = timetable[day_name]

    # Get Google services
    print("\nAuthenticating with Google...")
    try:
        service = get_forms_service()
        drive_service = get_drive_service()
        print("✓ Authentication successful!\n")
    except FileNotFoundError:
        sys.exit(1)
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)

    # Verify access to the target Drive folder before creating anything
    if config.drive_folder_id:
        try:
            drive_service.files().get(
                fileId=config.drive_folder_id, fields="id", supportsAllDrives=True
            ).execute()
        except Exception as e:
            print(f"❌ Cannot access Drive folder '{config.drive_folder_id}': {e}")
            sys.exit(1)

    # Create form for the selected day
    print(f"Creating feedback form for: {config.camp_name} - {day_name}")

    form_id, form = create_daily_feedback_form(
        service, drive_service, config, day_name, day_number, day_config
    )

    # Summary
    print("=" * 60)
    print("Form created successfully!")
    print("=" * 60)
    print(f"Day {day_number} ({day_name}):")
    print(f"  Edit: https://docs.google.com/forms/d/{form_id}/edit")
    print(f"  Live: {form['responderUri']}")
    print()


if __name__ == "__main__":
    main()
