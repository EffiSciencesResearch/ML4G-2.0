# Template Duplicator

An Apps Script bound to the **New camp template duplicator** spreadsheet.
One click duplicates the bootcamp template tree, substitutes `{{VAR}}`
placeholders, and rewires links/permissions.

## What it does

Given the spreadsheet's **Template Variables** table:

| Variable name | Value |
|---|---|
| `LONG_NAME` | `ML4Good Germany 2026` |
| `_FOLDER_OR_DOC_TO_DUPLICATE` | (link to template folder) |
| `ANONYMOUS_FEEDBACK_FORM` | (hyperlinked text or smart chip to the form) |
| … | … |

Click **Template Duplicator → Duplicate template…** in the menu. The script:

1. Recursively copies the source folder/doc next to *this spreadsheet*.
2. Substitutes `{{VAR}}` in every **title** (folders, docs, slides, forms, sheets).
3. Substitutes `{{VAR}}` in **Doc** and **Slide bodies** and **Form internals**
   (title, description, items, choices, confirmation message). Sheets get titles
   only.
4. Rewrites in-tree URLs in the variables map so chips/links land on the new
   copies. Forms get their **respondent URL** (auto-published).
5. Re-targets Drive **shortcuts** to in-tree files when their target was copied.
6. Replays **non-inherited** Drive permissions onto each copy.
7. Surfaces unknown `{{NAME}}` placeholders (typo detection) in the final
   dialog and a **Run log** sheet.

Variables whose name starts with `_` are control vars and never substituted
into bodies (e.g. `_FOLDER_OR_DOC_TO_DUPLICATE`).

## How variables can be stored

In a Value cell, the form/file link can be:

- **Plain hyperlink** — type text, Ctrl+K, paste URL. Recommended for forms.
- **Smart chip** — `@` then pick a Drive file. Rendered as a Docs chip in
  copies (except forms, which always become hyperlinked text).
- **Plain text URL** — no link applied. Becomes plain text in the doc.

For all three, in-tree IDs are rewritten to the new copy, and form URLs are
converted to the respondent (`/forms/d/e/.../viewform`) URL.

## Setup (developer)

```sh
cd meta/template_duplicator
clasp push -f
```

First run from the menu will prompt for OAuth grants
(Sheets / Docs / Slides / Drive / Forms).

The bound `scriptId` is in `.clasp.json`; the host spreadsheet is
`1uAQI6Rwe72vOoROBNhsTgQbtlxR7wGnsQ_jaFt_QWVc`.

## Files

- `Main.js` — orchestration
- `Variables.js` — reads the Template Variables table + smart chips
- `Copy.js` — recursive Drive copy, idMap, URL rewriting, shortcut re-targeting
- `Replace.js` — body substitution for Docs / Slides / Forms
- `Chips.js` — inserts real Docs smart chips via the Docs REST API
- `Permissions.js` — replays non-inherited perms
- `Util.js` — toast, logging, retry-with-backoff helper
- `Menu.js` — menu wiring
- `Debug.js` — diagnostic commands (chip URIs, variable dump, etc.)

## When it doesn't work

- **`User rate limit exceeded`** — retried automatically (1s → 16s, jittered).
  If it still fails, re-run; copies already done won't be redone.
- **Placeholder left in a copy** — see the final dialog or the `Run log` sheet
  for the name; either a typo in a template, or in a Form item type we don't
  yet substitute (grid/scale/image).
- **Form copy not accepting responses** — check `Run log` for
  `form-setPublished` errors; the running user must own the copy.
