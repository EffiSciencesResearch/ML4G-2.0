#!/usr/bin/env python3
"""
Main script to create daily feedback forms for a multi-day camp.
"""

import sys
import os
from utils import (
    get_forms_service, 
    get_drive_service,
    load_config, 
    create_base_form, 
    add_questions_to_form,
    create_question_from_config,
    create_scale_question,
    create_text_question,
    create_paragraph_question,
    create_choice_question,
    upload_image_to_drive_and_get_url,
    create_image_item_with_url,
    create_form_shortcut_in_drive
)

def create_lecture_questions(sessions, teachers):
    """Create rating and feedback questions for each session."""
    questions = []
    
    for session in sessions:
        # Handle both old string format and new object format for backwards compatibility
        if isinstance(session, str):
            session_name = session
            is_reading_group = False
        else:
            session_name = session['name']
            is_reading_group = session.get('reading_group', False)
        
        # Mandatory rating question (1-5 scale)
        rating_question = create_scale_question(
            f"How would you rate the '{session_name}' session?",
            required=True
        )
        questions.append(rating_question)
        
        # If it's a reading group, add teacher facilitation question
        if is_reading_group:
            teacher_question = create_choice_question(
                f"Which teacher facilitated the '{session_name}' reading group?",
                teachers,
                required=False
            )
            questions.append(teacher_question)
        
        # Optional feedback question - USE PARAGRAPH instead of text
        feedback_question = create_paragraph_question(
            f"Any additional feedback on '{session_name}'?",
            required=False
        )
        questions.append(feedback_question)
    
    return questions

def create_daily_feedback_form(service, drive_service, config, day_name, day_number, day_config):
    """Create a feedback form for a specific day."""
    # Create form title
    form_title = f"{config['camp_name']} Day {day_number} Feedback Form"
    
    print(f"Creating form: {form_title}")
    
    # Create the base form with description
    description = config.get('form_description', '')
    form_id, form = create_base_form(service, form_title, description)
    
    # Extract sessions from day config (handle both old and new format)
    if isinstance(day_config, list):
        # Old format: day_config is directly the list of sessions
        sessions = day_config
        meme_filename = None
        day_questions = []
    else:
        # New format: day_config has 'sessions' and 'meme'
        sessions = day_config.get('sessions', [])
        meme_filename = day_config.get('meme')
        day_questions = day_config.get('day_questions', [])
    
    # Prepare all questions
    all_questions = []
    
    # Add pre-questions
    print(f"  Adding {len(config['pre_questions'])} pre-questions...")
    for q_config in config['pre_questions']:
        question = create_question_from_config(q_config)
        all_questions.append(question)
    
    # Add session-specific questions
    teachers = config.get('teachers', [])
    print(f"  Adding questions for {len(sessions)} sessions...")
    lecture_questions = create_lecture_questions(sessions, teachers)
    all_questions.extend(lecture_questions)
    
    # Add day-specific extra questions (if any)
    if day_questions:
        print(f"  Adding {len(day_questions)} day-specific questions...")
        for q_config in day_questions:
            question = create_question_from_config(q_config)
            all_questions.append(question)
    
    # Add post-questions
    print(f"  Adding {len(config['post_questions'])} post-questions...")
    for q_config in config['post_questions']:
        question = create_question_from_config(q_config)
        all_questions.append(question)
    
    # Add meme image and caption question if specified
    if meme_filename:
        print(f"  Adding meme: {meme_filename}")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        meme_path = os.path.join(script_dir, "memes", meme_filename)
        
        # Upload image to Drive and get URL
        image_url = upload_image_to_drive_and_get_url(drive_service, meme_path)
        print(f"    Uploaded to Drive: {image_url}")
        
        # Add meme image
        meme_image = create_image_item_with_url(image_url)
        all_questions.append(meme_image)
        
        # Add caption question
        caption_question = create_text_question(
            "Provide a caption for this meme that describes your day!",
            required=False
        )
        all_questions.append(caption_question)
    
    # Add all questions to the form
    add_questions_to_form(service, form_id, all_questions)
    
    print(f"  ✓ Form created successfully!")
    print(f"  Edit URL: https://docs.google.com/forms/d/{form_id}/edit")
    print(f"  Live URL: {form['responderUri']}")
    
    # Create shortcut in Drive folder if configured
    drive_folder_id = config.get('drive_folder_id', '').strip()
    if drive_folder_id:
        try:
            shortcut_id, shortcut_link = create_form_shortcut_in_drive(
                drive_service, form_id, form_title, drive_folder_id
            )
            print(f"  ✓ Shortcut created in Drive folder")
        except Exception as e:
            print(f"  ⚠ Failed to create Drive shortcut: {e}")
    
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
    timetable = config['timetable']
    print(f"Available days:")
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
    service = get_forms_service()
    drive_service = get_drive_service()
    print("✓ Authentication successful!\n")
    
    # Create form for the selected day
    camp_name = config['camp_name']
    print(f"Creating feedback form for: {camp_name} - {day_name}")
    
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