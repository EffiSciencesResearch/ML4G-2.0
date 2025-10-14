
/**
 * Updates column headers starting with "D0", "D1", "D2", etc.
 * Prompts the user for a start date for D0.
 * Reformats the headers to "D# - DayOfWeek DayOfMonth Month Year" (e.g., "D0 - Mon 24 March 2025").
 * Operates on the active sheet.
 */
function updateDayHeadersWithDates() {
  const ui = SpreadsheetApp.getUi();
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = ss.getActiveSheet();

  if (!sheet) {
    ui.alert("No active sheet found. Please select a sheet first.");
    return;
  }

  // 1. Prompt user for the start date (for D0)
  const response = ui.prompt(
    'Enter Start Date for D0',
    'Please enter the date for D0 (e.g., YYYY-MM-DD, MM/DD/YYYY, or "July 20 2024"):',
    ui.ButtonSet.OK_CANCEL
  );

  if (response.getSelectedButton() !== ui.Button.OK) {
    ui.alert('Operation cancelled by user.');
    return;
  }

  const startDateString = response.getResponseText();
  if (!startDateString) {
    ui.alert('No date entered. Operation cancelled.');
    return;
  }

  const startDate = new Date(startDateString);
  // Check if the date is valid
  if (isNaN(startDate.getTime())) {
    ui.alert('Invalid Date', 'The date you entered ("' + startDateString + '") could not be understood. Please use a recognizable format (e.g., YYYY-MM-DD, MM/DD/YYYY, or "Month Day Year").', ui.ButtonSet.OK);
    return;
  }

  // 2. Identify and update headers
  const headerRowToUpdate = 1; // Assuming headers are in the first row (use your HEADER_ROW constant if defined globally)
  const lastColumn = sheet.getLastColumn();
  if (lastColumn === 0) {
    ui.alert("Sheet appears to be empty.");
    return;
  }

  const headerRange = sheet.getRange(headerRowToUpdate, 1, 1, lastColumn);
  const headerValues = headerRange.getValues()[0]; // Get a 1D array of header values
  let changesMade = 0;

  for (let i = 0; i < headerValues.length; i++) {
    let currentHeader = String(headerValues[i]).trim(); // Convert to string and trim whitespace

    // Regex to match "D" followed by one or more digits, at the start of the string.
    // This allows it to update headers that might already have some date info,
    // or are just "D0", "D1", etc.
    const match = currentHeader.match(/^D(\d+)/);

    if (match) {
      const dayNumber = parseInt(match[1], 10); // The number after "D" (e.g., 0 from "D0")

      // Calculate the date for this D-day
      let targetDate = new Date(startDate.getTime()); // Create a new Date object from startDate
      targetDate.setDate(startDate.getDate() + dayNumber); // Add the offset

      // Format the date: "DayOfWeek DayOfMonth Month Year"
      const dayOfWeek = targetDate.toLocaleDateString('en-US', { weekday: 'short' }); // e.g., "Mon"
      const dayOfMonth = targetDate.getDate(); // e.g., 24
      const month = targetDate.toLocaleDateString('en-US', { month: 'long' });    // e.g., "March"
      const year = targetDate.getFullYear(); // e.g., 2025

      // Construct the new header
      const newHeader = `D${dayNumber} - ${dayOfWeek} ${dayOfMonth} ${month} ${year}`;

      // Update the cell in the sheet
      sheet.getRange(headerRowToUpdate, i + 1).setValue(newHeader); // i+1 because columns are 1-indexed
      changesMade++;
    }
  }

  if (changesMade > 0) {
    ui.alert('Headers Updated', `${changesMade} day headers were updated successfully to the new format.`, ui.ButtonSet.OK);
  } else {
    ui.alert('No Matching Headers', 'No headers starting with "D" followed by a number (e.g., D0, D1) were found in the first row.', ui.ButtonSet.OK);
  }
}
