# Career Docs

Streamlit page that duplicates a Google Docs template once per participant, replacing `[NAME]` placeholders in the document body and filename, then shares each copy with its participant.

The page is `web.py`; it shows up as **Career Docs** in the [internal web app](../web/README.md) nav. No CLI.

Requires `service_account_token.json` at the repo root with view access to the template and write access to the destination Drive folder.
