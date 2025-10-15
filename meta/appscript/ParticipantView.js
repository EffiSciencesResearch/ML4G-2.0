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

  let targetSpreadsheet;
  const ui = SpreadsheetApp.getUi();
  let targetSheetName = null;

  if (targetUrl) {
    try {
      targetSheetName = SpreadsheetApp.openByUrl(targetUrl).getName();
    } catch (e) {
      ui.alert(`Failed to open the target spreadsheet URL: ${targetUrl}\n\nPlease check the URL and permissions.\nError: ${e.message}\n\nThe script will now proceed as if no target URL was found.`);
      targetUrl = null;
    }
  }

  if (targetUrl && targetSheetName) {
    const title = 'Select Output Destination';
    const prompt = `A target spreadsheet was found:\n"${targetSheetName}"\n\nDo you want to update it?\n\n- Click YES to update the target.\n- Click NO to create a new tab in this spreadsheet.`;
    const response = ui.alert(title, prompt, ui.ButtonSet.YES_NO_CANCEL);

    if (response === ui.Button.YES) {
      targetSpreadsheet = SpreadsheetApp.openByUrl(targetUrl);
    } else if (response === ui.Button.NO) {
      targetSpreadsheet = ss;
    } else {
      return; // User cancelled
    }
  } else {
    const title = 'Select Output Destination';
    const prompt = 'No target URL found.\nTo specify a target, add a cell containing exactly "Participant View Sheet: https://..."\n\nDo you want to create the participant view in a new tab in the current spreadsheet?';
    const response = ui.alert(title, prompt, ui.ButtonSet.OK_CANCEL);

    if (response === ui.Button.OK) {
      targetSpreadsheet = ss;
    } else {
      return; // User cancelled
    }
  }

  const sourceSheetName = sourceSheet.getName();
  let finalSheetName = sourceSheetName;
  if (targetSpreadsheet.getId() === ss.getId()) {
    finalSheetName = sourceSheetName + " - Participant View";
  }

  // Copy the sheet first. This ensures that even if we are replacing the *only*
  // sheet in the target, we won't fail by trying to delete the last sheet.
  const targetSheet = sourceSheet.copyTo(targetSpreadsheet);

  // Remove the old version of the sheet if it exists.
  let existingTargetSheet = targetSpreadsheet.getSheetByName(finalSheetName);
  if (existingTargetSheet) {
    targetSpreadsheet.deleteSheet(existingTargetSheet);
  }

  // Rename the newly copied sheet to the correct name.
  targetSheet.setName(finalSheetName);


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

  // To speed up the process, we group contiguous columns and delete them in batches.
  // We iterate from right to left to avoid shifting the indices of columns we still need to process.
  if (columnsToDelete.length > 0) {
    let i = columnsToDelete.length - 1;
    while (i >= 0) {
      let count = 1;
      let startColumn = columnsToDelete[i];
      // Find how many contiguous columns there are to delete from this point to the left.
      while (i > 0 && columnsToDelete[i - 1] === startColumn - 1) {
        count++;
        startColumn--;
        i--;
      }
      targetSheet.deleteColumns(startColumn, count);
      i--;
    }
  }

  // Protect the sheet to prevent accidental edits
  const protection = targetSheet.protect();
  protection.setDescription('This sheet is automatically generated. Please edit the source sheet, and rerun the script to update.');
  protection.setWarningOnly(true);

  SpreadsheetApp.getUi().alert(`Sheet "${finalSheetName}" in spreadsheet "${targetSpreadsheet.getName()}" created/updated successfully with ${columnsToCopyIndices.length} columns!`);
}
