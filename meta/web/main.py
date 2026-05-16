from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

META = Path(__file__).resolve().parent.parent

# Pages are registered explicitly: title, path. The path may point into any
# tool folder (vertical layout) or into meta/web/pages/ (frame pages like
# the dashboard and camp CRUD).
PAGES = [
    ("🏠 Dashboard", META / "web" / "pages" / "dashboard.py"),
    ("Career Docs", META / "career_docs" / "web.py"),
    ("Edit Camp", META / "web" / "pages" / "edit_camp.py"),
    ("Create Camp", META / "web" / "pages" / "create_camp.py"),
    ("One on One Scheduler", META / "one_on_ones" / "web.py"),
    ("Session Feedback", META / "web" / "pages" / "session_feedback.py"),
    ("Feedback Form Creator", META / "feedback_form_creator" / "web.py"),
]


def _url_path(path: Path) -> str:
    # Use the parent folder name for tool pages (meta/<tool>/web.py) and
    # the filename for frame pages (meta/web/pages/<name>.py).
    return path.parent.name if path.name == "web.py" else path.stem


page = st.navigation(
    [
        st.Page(path, title=title, url_path=_url_path(path), default=(i == 0))
        for i, (title, path) in enumerate(PAGES)
    ]
)
page.run()
