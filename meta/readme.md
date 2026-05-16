# ML4G Internal Tools

This folder contains all the internal tools we use to run ML4G bootcamps. Each subfolder is a self-contained tool with its own README; this page is just an index.

## Python tools

- **[Slack Message Scheduler](./slack_reminders/README.md)** — schedule Slack messages from CSV/YAML, export channel members, manage scheduled drops.
- **[Feedback Form Creator](./feedback_form_creator/README.md)** — generate the daily Google Forms feedback questionnaire for each day of a camp.
- **[Notebook Tools](./notebook_tools/README.md)** — workshop-authoring CLI: clean + badge + format `.ipynb` files, generate exercise notebooks from solutions, check links, fix typos.
- **[Drive Changelog](./drive_changelog/README.md)** — currently unused, kept around.

## Internal web app

- **[Streamlit dashboard](./web/README.md)** — internal UI for camp creation, career-doc duplication, feedback analysis, one-on-one scheduling.

## Apps Script tools

- **[Timetable Tools](./timetable_tools/readme.md)** — Bootcamp Tools menu added to the timetable Google Sheet (participant view, day-header dating).
- **[Bootcamp Template Duplicator](./template_duplicator/README.md)** — one-click duplication of the whole bootcamp Drive template tree.

## Getting help

- Each tool's README is the source of truth for how to run it. Start there.
- Ask Diego on Slack if you get stuck.
