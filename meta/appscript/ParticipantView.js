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

  const sourceSheetName = sourceSheet.getName();
  const targetSheetName = `${sourceSheetName} - participants`;

  // Remove old summary sheet if it exists, then create a new one
  let targetSheet = ss.getSheetByName(targetSheetName);
  if (targetSheet) {
    ss.deleteSheet(targetSheet);
  }
  targetSheet = ss.insertSheet(targetSheetName, ss.getSheets().length); // Insert as the last sheet

  const headerRow = 1;
  const lastSourceColumn = sourceSheet.getLastColumn();
  const columnsToCopyIndices = []; // Will store 1-based indices of columns to copy

  // 1. Identify columns with bold headers
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
    targetSheet.getRange("A1").setValue("No columns with bold headers found in '" + sourceSheetName + "'.");
    return;
  }

  // 2. Copy identified columns
  let currentTargetColumnIndex = 1; // Start pasting into column A (index 1) of the target sheet

  columnsToCopyIndices.forEach(sourceColIndex => {
    // Determine the last row with content in this specific source column
    let lastRowInSourceColumn = sourceSheet.getMaxRows(); // Start with max rows
    const columnValues = sourceSheet.getRange(1, sourceColIndex, sourceSheet.getMaxRows(), 1).getValues();
    for (let r = columnValues.length - 1; r >= 0; r--) {
      if (columnValues[r][0] !== "") {
        lastRowInSourceColumn = r + 1;
        break;
      }
    }
    if (lastRowInSourceColumn === 0 && sourceSheet.getMaxRows() > 0) lastRowInSourceColumn = 1; // At least copy the header if column is truly empty


    const sourceRange = sourceSheet.getRange(1, sourceColIndex, lastRowInSourceColumn, 1); // (startRow, startCol, numRows, numCols)
    const targetCell = targetSheet.getRange(1, currentTargetColumnIndex); // Top-left cell of the target column

    // PASTE_NORMAL copies values, formatting, data validation, merged cells, etc.
    sourceRange.copyTo(targetCell, SpreadsheetApp.CopyPasteType.PASTE_NORMAL, false);

    // Optional: Adjust column width in target sheet to match source
    const sourceColumnWidth = sourceSheet.getColumnWidth(sourceColIndex);
    targetSheet.setColumnWidth(currentTargetColumnIndex, sourceColumnWidth);

    currentTargetColumnIndex++;
  });

  SpreadsheetApp.getUi().alert(`Sheet "${targetSheetName}" created/updated successfully with ${columnsToCopyIndices.length} columns!`);
}
