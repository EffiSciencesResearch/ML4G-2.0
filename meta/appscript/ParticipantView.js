/**
 * Creates a new sheet named "{Original Sheet Name} - participants"
 * and copies only the columns from the active sheet whose header (row 1) is bold.
 * It preserves formatting, merged cells, and colors.
 */
function createParticipantViewFromActiveSheet() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sourceSheet = ss.getActiveSheet();

  if (!sourceSheet) {
    SpreadsheetApp.getUi().alert("No active sheet found. Please select a sheet first.");
    return;
  }

  // 1. Find the cell with "Participant View Sheet: <URL>"
  const dataRange = sourceSheet.getDataRange();
  const values = dataRange.getValues();
  let targetUrl = null;
  const searchString = "Participant View Sheet:";

  for (let i = 0; i < values.length; i++) {
    for (let j = 0; j < values[i].length; j++) {
      if (typeof values[i][j] === 'string' && values[i][j].includes(searchString)) {
        targetUrl = values[i][j].substring(values[i][j].indexOf(searchString) + searchString.length).trim();
        break;
      }
    }
    if (targetUrl) {
      break;
    }
  }

  if (!targetUrl) {
    SpreadsheetApp.getUi().alert(`Could not find a cell in sheet '${sourceSheet.getName()}' containing '${searchString} <URL>'.`);
    return;
  }

  let targetSpreadsheet;
  try {
    targetSpreadsheet = SpreadsheetApp.openByUrl(targetUrl);
  } catch (e) {
    SpreadsheetApp.getUi().alert("Failed to open the target spreadsheet. Please check the URL and permissions.\nError: " + e.message);
    return;
  }

  const sourceSheetName = sourceSheet.getName();

  // Remove old summary sheet if it exists
  let existingTargetSheet = targetSpreadsheet.getSheetByName(sourceSheetName);
  if (existingTargetSheet) {
    targetSpreadsheet.deleteSheet(existingTargetSheet);
  }

  // Copy the entire sheet to the target spreadsheet
  const targetSheet = sourceSheet.copyTo(targetSpreadsheet);
  targetSheet.setName(sourceSheetName);


  const headerRow = 1;
  const lastSourceColumn = sourceSheet.getLastColumn();
  const columnsToCopyIndices = []; // Will store 1-based indices of columns to copy

  // 2. Identify columns with bold headers
  if (lastSourceColumn > 0) {
    const headerRange = sourceSheet.getRange(headerRow, 1, 1, lastSourceColumn);
    const fontWeights = headerRange.getFontWeights()[0]; // getFontWeights returns 2D array, we need the first row

    for (let i = 0; i < fontWeights.length; i++) {
      if (fontWeights[i] === "bold") {
        columnsToCopyIndices.push(i + 1); // Column indices are 1-based
      }
    }
  }

  if (columnsToCopyIndices.length === 0) {
    SpreadsheetApp.getUi().alert("No columns with bold headers found in the active sheet.");
    targetSheet.clear(); // Clear the copied sheet
    targetSheet.getRange("A1").setValue("No columns with bold headers found in '" + sourceSheetName + "'.");
    return;
  }

  // 3. Delete columns that are NOT in columnsToCopyIndices from the new sheet
  const lastTargetColumn = targetSheet.getLastColumn();
  const columnsToDelete = [];
  for (let i = 1; i <= lastTargetColumn; i++) {
    if (columnsToCopyIndices.indexOf(i) === -1) {
      columnsToDelete.push(i);
    }
  }

  // Delete from right to left to avoid index shifting
  for (let i = columnsToDelete.length - 1; i >= 0; i--) {
    targetSheet.deleteColumn(columnsToDelete[i]);
  }

  SpreadsheetApp.getUi().alert(`Sheet "${sourceSheetName}" in spreadsheet "${targetSpreadsheet.getName()}" created/updated successfully with ${columnsToCopyIndices.length} columns!`);
}
