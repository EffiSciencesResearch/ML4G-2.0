# ML4G Automation Tools - User Guide

This folder contains helpful scripts to automate common tasks. No programming experience needed!

---

## Table of Contents
1. [Slack Message Scheduler](./slack_reminders/README.md) - Schedule Slack messages automatically
2. [Google Ops](./google_ops/README.md) - Drive duplication, prefix renaming, per-person doc copies
3. [Bootcamp Template Duplicator](#bootcamp-template-duplicator) - One-click duplicate the whole template tree for a new camp

---

## Bootcamp Template Duplicator

**What it does:** Duplicates the whole bootcamp Drive template tree (folders,
docs, slides, sheets, forms) in one click, with `{{VAR}}` substitution in
titles and bodies, link rewriting, and permission replay. Designed to replace
the manual ~20-times-a-year cloning ritual.

Unlike the Python "Google Docs Duplicator" above (which makes per-person
copies of a single doc), this one duplicates a whole tree for a new camp.

### How to use it

Open the **New camp template duplicator** Google Sheet, then:

1. Fill in the **Template Variables** table — at minimum `LONG_NAME` and
   `_FOLDER_OR_DOC_TO_DUPLICATE` (which points to the template root folder).
2. Menu: **Template Duplicator → Duplicate template…**
3. First time only, grant the OAuth scopes it asks for.
4. Wait for the toast notifications. The final dialog gives you a link to
   the new folder and flags any unknown `{{NAME}}` placeholders it found.

The new tree is placed as a sibling of the duplicator spreadsheet itself, so
keep that spreadsheet inside the year's parent folder.

### When something looks off

- Open the **Run log** sheet inside the duplicator spreadsheet — every error,
  retry, and unknown-placeholder gets a row.
- For a fresh diagnostic, the menu has `Debug: chip URIs` and
  `Debug: dump variables`.

### Developer notes

Source lives in [`meta/template_duplicator/`](./template_duplicator/) and is
deployed via `clasp push -f`. See its [README](./template_duplicator/README.md)
for the file layout and how the substitution pipeline works.

---

## Getting Help

- Check the full documentation in each Python file for advanced options
- For Slack message scheduling, see [slack_reminders/README.md](./slack_reminders/README.md)
- Ask Diego if you get stuck!
