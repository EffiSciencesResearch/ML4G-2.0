// CALENDAR -------------
// --- CONFIGURATION ---
// const EVENT_ID_PREFIX = 'bootcampTimetableEvent_'; // No longer primary for deletion, but could be kept for other ID purposes
const HEADER_ROW = 1; // Row number for the main day headers AND TA initial headers
const DATA_START_ROW = 3; // Row number where actual event data begins
const START_TIME_COLUMN = 1; // Column A (1-indexed) for event start times
const END_TIME_COLUMN = 2;   // Column B (1-indexed) for event end times
const CALENDAR_EVENT_FETCH_RANGE_YEARS = 5; // How many years past/future to scan for events to delete

// --- DESCRIPTION MARKER ---
// This marker will be embedded in the event description to identify events created by this script from this specific sheet.
const SCRIPT_GENERATED_EVENT_MARKER_PREFIX = "\n\n[TimetableSourceSheetID:"; // Add newlines for better readability in description
const SCRIPT_GENERATED_EVENT_MARKER_SUFFIX = "]";


function promptForCalendarSettings_UI_V2() {
  const ui = SpreadsheetApp.getUi();

  // 1. Get TA Initial
  const taResponse = ui.prompt(
    'Filter by TA',
    'Enter TA initial (e.g., D, J, L, T) or leave blank/type "ALL" for all sessions:',
    ui.ButtonSet.OK_CANCEL
  );
  if (taResponse.getSelectedButton() !== ui.Button.OK) return;
  let taFilter = taResponse.getResponseText().trim().toUpperCase();
  if (taFilter === "ALL" || taFilter === "") {
    taFilter = "ALL"; // Standardize
  }

  // 2. Get Target Calendar
  const calendars = CalendarApp.getAllCalendars();
  let calendarNames = [];
  if (calendars && calendars.length > 0) {
      calendarNames = calendars.map(cal => cal.getName());
  }
  calendarNames.unshift("[Create New Calendar]"); // Add option to create new

  const calChoiceResponse = ui.prompt(
    'Select Calendar',
    'Choose a calendar to add sessions to, or create a new one:\n\n' + calendarNames.map((name, i) => `${i + 1}. ${name}`).join('\n'),
    ui.ButtonSet.OK_CANCEL
  );
  if (calChoiceResponse.getSelectedButton() !== ui.Button.OK) return;

  const choiceText = calChoiceResponse.getResponseText();
  const choiceIndex = parseInt(choiceText, 10) - 1;
  let targetCalendar;
  let newCalNameFromUser = "";

  if (isNaN(choiceIndex) || choiceIndex < 0 || choiceIndex >= calendarNames.length) {
     // Allow typing the new calendar name directly if "[Create New Calendar]" is the only option and text was entered
     if (calendarNames.length === 1 && calendarNames[0] === "[Create New Calendar]" && choiceText) {
        newCalNameFromUser = choiceText.trim();
     } else {
        ui.alert("Invalid calendar choice.");
        return;
     }
  }

  if ((choiceIndex === 0 && calendarNames[0] === "[Create New Calendar]") || newCalNameFromUser) {
    let promptedCalName = newCalNameFromUser;
    if (!promptedCalName) { // If not directly entered, prompt for new name
        const newCalNameResponse = ui.prompt(
            'New Calendar Name',
            'Enter the name for the new calendar:',
            ui.ButtonSet.OK_CANCEL
        );
        if (newCalNameResponse.getSelectedButton() !== ui.Button.OK || !newCalNameResponse.getResponseText()) {
            ui.alert("Calendar creation cancelled or no name provided.");
            return;
        }
        promptedCalName = newCalNameResponse.getResponseText().trim();
    }

    try {
      targetCalendar = CalendarApp.createCalendar(promptedCalName);
      ui.alert('Calendar Created', `New calendar "${promptedCalName}" created successfully. Events will be added to it.`, ui.ButtonSet.OK);
    } catch (e) {
      ui.alert('Error', `Could not create new calendar "${promptedCalName}": ${e.toString()}`, ui.ButtonSet.OK);
      return;
    }
  } else {
    targetCalendar = calendars[choiceIndex - 1]; // Adjust because we unshifted
  }

  if (!targetCalendar) {
    ui.alert("Failed to select or create a calendar.");
    return;
  }

  const result = addSessionsToCalendar(targetCalendar.getId(), taFilter);
  ui.alert(result.title, result.message, ui.ButtonSet.OK);
}

function addSessionsToCalendar(calendarId, taFilter) {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  const calendar = CalendarApp.getCalendarById(calendarId);
  const scriptTimeZone = Session.getScriptTimeZone();
  const sheetId = sheet.getSheetId(); // Unique ID for the current sheet

  if (!calendar) {
    Logger.log(`Error: Target calendar with ID "${calendarId}" not found.`);
    return { title: 'Error', message: 'Target calendar not found.' };
  }
  Logger.log(`Using calendar: "${calendar.getName()}", Timezone: ${calendar.getTimeZone()}. Script timezone: ${scriptTimeZone}`);

  let eventsCleared = 0;
  // Construct the unique marker to find events from THIS sheet
  const sheetSpecificDescriptionMarker = `${SCRIPT_GENERATED_EVENT_MARKER_PREFIX}${sheetId}${SCRIPT_GENERATED_EVENT_MARKER_SUFFIX}`;

  Logger.log(`Attempting to clear pre-existing events from this sheet (searching for description marker: "${sheetSpecificDescriptionMarker}")...`);
  try {
    const today = new Date();
    const startDateFetch = new Date(today.getFullYear() - CALENDAR_EVENT_FETCH_RANGE_YEARS, 0, 1);
    const endDateFetch = new Date(today.getFullYear() + CALENDAR_EVENT_FETCH_RANGE_YEARS, 11, 31);
    const existingEvents = calendar.getEvents(startDateFetch, endDateFetch);

    for (const eventToCheck of existingEvents) {
      try {
        if (eventToCheck && typeof eventToCheck.getDescription === 'function') {
          const description = eventToCheck.getDescription();
          if (description && description.includes(sheetSpecificDescriptionMarker)) {
            eventToCheck.deleteEvent();
            eventsCleared++;
          }
        }
      } catch (e) {
        Logger.log(`  Error processing an existing event for deletion (description check): ${e.toString()}. Event title: ${eventToCheck ? eventToCheck.getTitle() : 'N/A'}`);
      }
    }
    Logger.log(`${eventsCleared} event(s) cleared from this sheet.`);
  } catch (e) {
    Logger.log(`Error during event clearing phase: ${e.toString()}`);
  }

  const allValues = sheet.getDataRange().getValues();
  const lastRowInSheet = sheet.getLastRow();
  const lastColInSheet = sheet.getLastColumn();
  let eventsAdded = 0;

  const headerRowValues = (HEADER_ROW -1 < allValues.length) ? allValues[HEADER_ROW - 1] : [];

  const taAssignmentColumnIndices = [];
  for (let c = 0; c < headerRowValues.length && c < lastColInSheet; c++) {
      const headerVal = String(headerRowValues[c]).trim();
      if (headerVal.length === 1 && headerVal >= 'A' && headerVal <= 'Z') {
          taAssignmentColumnIndices.push(c);
      }
  }

  const dayColumnMainIndices = [];
  for (let c_header = 0; c_header < headerRowValues.length && c_header < lastColInSheet; c_header++) {
      if (String(headerRowValues[c_header]).trim().match(/^D(\d+)\s*-\s*(.+)/i)) {
          dayColumnMainIndices.push(c_header);
      }
  }

  for (let dayColIdx_loop = 0; dayColIdx_loop < dayColumnMainIndices.length; dayColIdx_loop++) {
    const dayColumnIndex_0based = dayColumnMainIndices[dayColIdx_loop];
    const headerText = String(headerRowValues[dayColumnIndex_0based]).trim();
    const dayMatch = headerText.match(/^D(\d+)\s*-\s*(.+)/i);
    if (!dayMatch) continue;

    const dateStringPart = dayMatch[2].trim();
    const currentDateBasis = new Date(dateStringPart);

    if (isNaN(currentDateBasis.getTime())) {
      Logger.log(`Skipping column ${dayColumnIndex_0based + 1} due to invalid date in header: "${headerText}" (parsed as "${dateStringPart}")`);
      continue;
    }

    let nextDayMainColumnIndex = lastColInSheet;
    if (dayColIdx_loop + 1 < dayColumnMainIndices.length) {
        nextDayMainColumnIndex = dayColumnMainIndices[dayColIdx_loop + 1];
    }

    for (let r_0based = DATA_START_ROW - 1; r_0based < lastRowInSheet; r_0based++) {
      const eventCellRow_1based = r_0based + 1;

      if (!allValues[r_0based] || dayColumnIndex_0based >= allValues[r_0based].length) {
          continue;
      }

      const eventCellRange = sheet.getRange(eventCellRow_1based, dayColumnIndex_0based + 1); // 1-based column

      if (eventCellRange.isPartOfMerge()) {
          const mergedRanges = eventCellRange.getMergedRanges();
          if (mergedRanges && mergedRanges.length > 0) {
              const mainMergeCell = mergedRanges[0].getCell(1,1);
              if (mainMergeCell.getRow() !== eventCellRow_1based || mainMergeCell.getColumn() !== (dayColumnIndex_0based + 1)) {
                  continue;
              }
          }
      }

      const eventTitle = String(allValues[r_0based][dayColumnIndex_0based]).trim();
      if (!eventTitle) {
          continue;
      }

      let timeStartRow_0based = r_0based;
      let timeEndRow_0based = r_0based;

      if (eventCellRange.isPartOfMerge()) {
          const mergedRange = eventCellRange.getMergedRanges()[0];
          timeStartRow_0based = mergedRange.getRow() - 1;
          timeEndRow_0based = mergedRange.getLastRow() - 1;
      }

      if (timeStartRow_0based >= allValues.length || !allValues[timeStartRow_0based] ||
          timeEndRow_0based >= allValues.length || !allValues[timeEndRow_0based]) {
          Logger.log(`Skipping event "${eventTitle}" (Sheet Row ${eventCellRow_1based}) due to time rows (${timeStartRow_0based+1}-${timeEndRow_0based+1}) being out of fetched data bounds.`);
          continue;
      }

      const startTimeStr = String(allValues[timeStartRow_0based][START_TIME_COLUMN - 1]);
      const endTimeStr = String(allValues[timeEndRow_0based][END_TIME_COLUMN - 1]);

      if (!startTimeStr || !endTimeStr) {
        Logger.log(`Skipping event "${eventTitle}" (Sheet Row ${eventCellRow_1based}) due to missing start or end time from effective rows ${timeStartRow_0based+1}-${timeEndRow_0based+1}.`);
        continue;
      }

      const sessionStartTime = combineDateAndTime(currentDateBasis, startTimeStr);
      const sessionEndTime = combineDateAndTime(currentDateBasis, endTimeStr);

      if (!sessionStartTime || !sessionEndTime) {
        Logger.log(`Skipping event "${eventTitle}" (Sheet Row ${eventCellRow_1based}) due to invalid time format for "${startTimeStr}" or "${endTimeStr}".`);
        continue;
      }

      if (sessionStartTime.getTime() >= sessionEndTime.getTime()) {
        Logger.log(`Skipping event "${eventTitle}" (Sheet Row ${eventCellRow_1based}) because start time is not before end time. Start: ${sessionStartTime.toISOString()}, End: ${sessionEndTime.toISOString()}`);
        continue;
      }

      let isTAAssignedForEvent = false;
      if (taFilter === "ALL") {
        isTAAssignedForEvent = true;
      } else {
        for (let r_ta_check = timeStartRow_0based; r_ta_check <= timeEndRow_0based; r_ta_check++) {
          if (r_ta_check >= allValues.length || !allValues[r_ta_check]) break;

          for (const taColIdx of taAssignmentColumnIndices) {
            if (taColIdx > dayColumnIndex_0based && taColIdx < nextDayMainColumnIndex) {
              if (taColIdx < allValues[r_ta_check].length &&
                  allValues[r_ta_check][taColIdx] &&
                  String(allValues[r_ta_check][taColIdx]).trim().toUpperCase() === taFilter) {
                isTAAssignedForEvent = true;
                break;
              }
            }
          }
          if (isTAAssignedForEvent) break;
        }
      }

      if (isTAAssignedForEvent) {
        // Add the unique sheet marker to the description
        const eventDescription = `Created by Bootcamp Timetable script.${sheetSpecificDescriptionMarker}`;

        try {
          const newCalEvent = calendar.createEvent(eventTitle, sessionStartTime, sessionEndTime, {
            description: eventDescription, // Set the description
            timeZone: scriptTimeZone
          });

          if (newCalEvent) { // Check if event object was returned
              eventsAdded++;
          } else {
              Logger.log(`Warning: Failed to obtain a CalendarEvent object for "${eventTitle}" (Sheet Row ${eventCellRow_1based}, Col ${dayColumnIndex_0based + 1}). Event might not have been created or an issue occurred.`);
          }
        } catch (e) {
          Logger.log(`Critical Error creating calendar event for "${eventTitle}" (Sheet Row ${eventCellRow_1based}, Col ${dayColumnIndex_0based + 1}): ${e.toString()}. Check parameters: Start=${sessionStartTime}, End=${sessionEndTime}.`);
        }
      }
    }
  }
  Logger.log(`Total events added in this run: ${eventsAdded}`);
  return {
    title: 'Calendar Update Complete',
    message: `${eventsCleared} old session(s) from this sheet cleared. ${eventsAdded} new session(s) added.`
  };
}

function combineDateAndTime(dateBasisObj, timeString) {
  const timeParts = String(timeString).match(/^(\d{1,2}):(\d{2})$/);
  if (!timeParts) {
    return null;
  }
  const hours = parseInt(timeParts[1], 10);
  const minutes = parseInt(timeParts[2], 10);
  if (hours > 23 || minutes > 59) {
    return null;
  }
  const year = dateBasisObj.getFullYear();
  const month = dateBasisObj.getMonth();
  const day = dateBasisObj.getDate();
  return new Date(year, month, day, hours, minutes, 0, 0);
}

function onInstall(e) {
  onOpen(e);
}

function onOpen(e) {
  const ui = SpreadsheetApp.getUi();
  const menu = ui.createMenu('Bootcamp Tools');
  menu.addItem('Create Participant View...', 'createParticipantViewFromActiveSheet');
  menu.addItem('Update Day Headers with Dates', 'updateDayHeadersWithDates');
  menu.addItem('Add/Update Sessions in Google Calendar...', 'promptForCalendarSettings_UI_V2')
  menu.addToUi();
}
