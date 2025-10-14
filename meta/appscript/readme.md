# Google Sheets helpers

This folder contains a few appscripts that we use to automate tasks in Google Sheets.

## Setup

1. You need to install [CLASP](https://developers.google.com/apps-script/guides/clasp), possibly using `npm install -g @google/clasp`
2. Login using `clasp login`

## Usage

To add the scripts to a Google Sheet:
1. Open the Google Sheet you want to add the scripts to
2. Go to `Extensions` -> `Apps Script`
3. It will open the Apps Script editor. Go to `Settings`.
4. Copy the script ID there.
5. Put it in the `.clasp.json` file.
6. Run `clasp push` to push the scripts to the Google Sheet.
7. Reload the Timetable, you should see `Bootcamp Tools` in the menu.
