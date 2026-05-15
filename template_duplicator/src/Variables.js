/**
 * Variable record:
 *   {
 *     name: string,
 *     plain: string,             // plain text of the Value cell
 *     cellLinkUrl: string|null,  // hyperlink applied to the whole Value cell, or chip URI
 *     mimeType: string|null,     // set when source cell was a Sheets smart chip
 *   }
 */

var VARIABLE_NAME_RE = /^\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}$/;
var PLACEHOLDER_RE = /\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}/g;

function loadVariables(ss) {
  // Locate the variables block by finding the header row
  // "Variable name | Value | Comment" anywhere in the spreadsheet.
  // (Sheet.getTables() is not available on the current Apps Script runtime.)
  var sheets = ss.getSheets();
  var found = null;
  for (var i = 0; i < sheets.length && !found; i++) {
    found = findHeaderInSheet_(sheets[i]);
  }
  if (!found) {
    throw new Error('Could not find a header row "Variable name | Value | Comment" in any sheet.');
  }

  var sheet = found.sheet;
  var headerRow = found.row;
  var nameCol = found.nameCol;
  var valueCol = found.valueCol;

  // Read downward from headerRow+1 until the Variable name cell is empty.
  var lastRow = sheet.getLastRow();
  var lastCol = Math.max(nameCol, valueCol);
  var height = lastRow - headerRow;
  if (height <= 0) return [];

  var dataRange = sheet.getRange(headerRow + 1, 1, height, lastCol);
  var values = dataRange.getValues();
  var rich = dataRange.getRichTextValues();

  // Read smart chips for the data range via the Sheets Advanced Service.
  // Returns chips[row][col] = { uri, mimeType, title } or null.
  var chips = loadChips_(ss.getId(), sheet, headerRow + 1, height, lastCol);

  var vars = [];
  for (var r = 0; r < values.length; r++) {
    var name = String(values[r][nameCol - 1] || '').trim();
    if (!name) break; // first empty Variable name ends the table
    var rv = rich[r][valueCol - 1];
    var chip = chips[r][valueCol - 1];

    var plain = String(values[r][valueCol - 1] == null ? '' : values[r][valueCol - 1]);
    var cellLinkUrl = rv ? rv.getLinkUrl() : null;
    var mimeType = null;

    if (chip) {
      // Smart chip in Value cell — use the chip's URL as the link.
      cellLinkUrl = chip.uri;
      mimeType = chip.mimeType || null;
      // If the cell had no fallback text (just a chip), use the chip's title.
      if (!plain && chip.title) plain = String(chip.title);
    }

    vars.push({
      name: name,
      plain: plain,
      cellLinkUrl: cellLinkUrl,
      mimeType: mimeType,
    });
  }
  return vars;
}

function loadChips_(spreadsheetId, sheet, startRow, height, width) {
  // 2-D array of null filling our data range.
  var out = [];
  for (var r = 0; r < height; r++) {
    var row = [];
    for (var c = 0; c < width; c++) row.push(null);
    out.push(row);
  }

  try {
    var a1 = "'" + sheet.getName().replace(/'/g, "''") + "'!" +
      sheet.getRange(startRow, 1, height, width).getA1Notation();
    var resp = Sheets.Spreadsheets.get(spreadsheetId, {
      ranges: [a1],
      includeGridData: true,
      fields: 'sheets.data.rowData.values.chipRuns',
    });
    var sheetsData = resp.sheets || [];
    if (!sheetsData.length) return out;
    var data = (sheetsData[0].data || [])[0];
    if (!data || !data.rowData) return out;

    for (var r2 = 0; r2 < data.rowData.length; r2++) {
      var rowVals = data.rowData[r2].values || [];
      for (var c2 = 0; c2 < rowVals.length; c2++) {
        var cellVal = rowVals[c2];
        if (cellVal && cellVal.chipRuns && cellVal.chipRuns.length) {
          var rl = cellVal.chipRuns[0].chip && cellVal.chipRuns[0].chip.richLinkProperties;
          if (rl && rl.uri) {
            out[r2][c2] = { uri: rl.uri, mimeType: rl.mimeType || null, title: rl.title || null };
          }
        }
      }
    }
  } catch (e) {
    console.warn('Sheets chipRuns fetch failed (continuing without chips): ' + (e && e.message));
  }
  return out;
}

function findHeaderInSheet_(sheet) {
  var lastRow = sheet.getLastRow();
  var lastCol = sheet.getLastColumn();
  if (lastRow === 0 || lastCol === 0) return null;
  var scanRows = Math.min(lastRow, 50);
  var values = sheet.getRange(1, 1, scanRows, lastCol).getValues();
  for (var r = 0; r < values.length; r++) {
    var row = values[r];
    var nameCol = -1, valueCol = -1;
    for (var c = 0; c < row.length; c++) {
      var cell = String(row[c]).trim();
      if (cell === 'Variable name') nameCol = c + 1;
      else if (cell === 'Value') valueCol = c + 1;
    }
    if (nameCol > 0 && valueCol > 0) {
      return { sheet: sheet, row: r + 1, nameCol: nameCol, valueCol: valueCol };
    }
  }
  return null;
}

/** Returns the variable named `name`, or null. Skips control vars (those starting with `_`). */
function findVar(vars, name) {
  for (var i = 0; i < vars.length; i++) if (vars[i].name === name) return vars[i];
  return null;
}

/** Resolves nested {{VAR}} references inside plain values until fixed point. */
function resolveNestedPlain(vars) {
  for (var iter = 0; iter < 10; iter++) {
    var changed = false;
    for (var i = 0; i < vars.length; i++) {
      var v = vars[i];
      var resolved = v.plain.replace(PLACEHOLDER_RE, function (_, n) {
        var ref = findVar(vars, n);
        return ref ? ref.plain : ('{{' + n + '}}');
      });
      if (resolved !== v.plain) { v.plain = resolved; changed = true; }
    }
    if (!changed) return;
  }
}

/** Substitute {{VAR}} in a plain-text title. */
function substituteTitle(title, vars) {
  return String(title).replace(PLACEHOLDER_RE, function (_, n) {
    var v = findVar(vars, n);
    return v ? v.plain : ('{{' + n + '}}');
  });
}
