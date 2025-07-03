from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

page_directory = (Path(__file__).parent / "webpages").resolve()

pages = [
    ("🏠 Dashboard", "dashboard.py"),
    ("Career Docs", "career_docs.py"),
    ("Edit Camp", "edit_camp.py"),
]

# Add pages not in the list above
already_listed = [p[1] for p in pages]
for file in page_directory.glob("*.py"):
    if file.name not in already_listed:
        pages.append((file.stem.replace("_", " ").title(), file.name))


page = st.navigation(
    [
        st.Page(
            page_directory / p,
            title=title,
            default=(i == 0),
        )
        for i, (title, p) in enumerate(pages)
    ]
)
page.run()
