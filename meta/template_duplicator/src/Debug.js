/**
 * Run this from the Apps Script editor (Run ▸ debugListTables) and check the
 * execution log. It dumps everything that could plausibly be the "Template
 * Variables" table so we can see what the API actually exposes.
 */
function debugListTables() {
  var ss = SpreadsheetApp.getActive();
  var log = [];
  log.push('Spreadsheet: ' + ss.getName() + '  id=' + ss.getId());

  // 1) Spreadsheet-level getTables(), if it exists.
  if (typeof ss.getTables === 'function') {
    try {
      var ssTables = ss.getTables();
      log.push('Spreadsheet.getTables() -> ' + ssTables.length + ' table(s)');
      ssTables.forEach(function (t, i) {
        log.push('  [' + i + '] name="' + safeCall_(t, 'getName') + '"  range=' + safeRange_(t));
      });
    } catch (e) {
      log.push('Spreadsheet.getTables() threw: ' + e.message);
    }
  } else {
    log.push('Spreadsheet.getTables is NOT a function on this runtime.');
  }

  // 2) Per-sheet inspection.
  var sheets = ss.getSheets();
  for (var s = 0; s < sheets.length; s++) {
    var sh = sheets[s];
    log.push('--- Sheet[' + s + '] name="' + sh.getName() + '" hidden=' + sh.isSheetHidden());

    // 2a) Sheet.getTables()
    if (typeof sh.getTables === 'function') {
      try {
        var tables = sh.getTables();
        log.push('  Sheet.getTables() -> ' + tables.length);
        tables.forEach(function (t, i) {
          log.push('    [' + i + '] name="' + safeCall_(t, 'getName') + '"  range=' + safeRange_(t));
        });
      } catch (e) {
        log.push('  Sheet.getTables() threw: ' + e.message);
      }
    } else {
      log.push('  Sheet.getTables is NOT a function.');
    }

    // 2b) Named ranges that overlap this sheet
    var named = ss.getNamedRanges();
    named.forEach(function (nr) {
      if (nr.getRange().getSheet().getName() === sh.getName()) {
        log.push('  named range: "' + nr.getName() + '" -> ' + nr.getRange().getA1Notation());
      }
    });

    // 2c) Bandings (formatted "tables")
    try {
      var bandings = sh.getBandings();
      bandings.forEach(function (b, i) {
        log.push('  banding[' + i + '] range=' + b.getRange().getA1Notation());
      });
    } catch (e) {}

    // 2d) Protected ranges
    try {
      var prots = sh.getProtections(SpreadsheetApp.ProtectionType.RANGE);
      prots.forEach(function (p) {
        log.push('  protected: "' + p.getDescription() + '" -> ' + p.getRange().getA1Notation());
      });
    } catch (e) {}

    // 2e) First non-empty row content (look for the header pattern manually)
    var last = sh.getLastRow();
    var lastCol = sh.getLastColumn();
    if (last > 0 && lastCol > 0) {
      var preview = sh.getRange(1, 1, Math.min(3, last), Math.min(6, lastCol)).getValues();
      preview.forEach(function (row, i) {
        log.push('  row[' + (i + 1) + ']: ' + JSON.stringify(row));
      });
    }
  }

  console.log(log.join('\n'));
  SpreadsheetApp.getUi().alert(
    'Wrote ' + log.length + ' debug lines to the execution log.\n\n' +
    'In the Apps Script editor: View ▸ Executions ▸ click the latest run ▸ read the log.'
  );
}

function safeCall_(obj, method) {
  try { return obj[method] ? String(obj[method]()) : '(no ' + method + ')'; }
  catch (e) { return '(throw: ' + e.message + ')'; }
}

function safeRange_(t) {
  try { return t.getRange ? t.getRange().getA1Notation() : '(no getRange)'; }
  catch (e) { return '(throw: ' + e.message + ')'; }
}

/**
 * Dump everything we can see about each cell in the variables table:
 * raw value, formula, RichTextValue text, cell-level link, and each run's
 * indices / style / link URL. Run from the menu, then open
 * View ▸ Executions in the Apps Script editor to read the log.
 */
function debugDumpVariables() {
  var ss = SpreadsheetApp.getActive();
  var sheets = ss.getSheets();
  var found = null;
  for (var i = 0; i < sheets.length && !found; i++) {
    found = findHeaderInSheet_(sheets[i]);
  }
  if (!found) { console.log('No header row found.'); return; }

  var sheet = found.sheet;
  var headerRow = found.row;
  var lastCol = sheet.getLastColumn();
  var lastRow = sheet.getLastRow();
  var height = lastRow - headerRow;

  var headerVals = sheet.getRange(headerRow, 1, 1, lastCol).getValues()[0];
  var dataRange = sheet.getRange(headerRow + 1, 1, height, lastCol);
  var values = dataRange.getValues();
  var formulas = dataRange.getFormulas();
  var rich = dataRange.getRichTextValues();

  var log = [];
  log.push('Sheet: "' + sheet.getName() + '"  header row: ' + headerRow);
  log.push('Header: ' + JSON.stringify(headerVals));
  log.push('Data range: ' + dataRange.getA1Notation());
  log.push('');

  for (var r = 0; r < values.length; r++) {
    var nameCell = String(values[r][found.nameCol - 1] || '').trim();
    if (!nameCell) { log.push('--- row ' + (headerRow + 1 + r) + ': (blank, stop) ---'); break; }
    log.push('=== Row ' + (headerRow + 1 + r) + ' ============================');
    for (var c = 0; c < lastCol; c++) {
      var col = c + 1;
      var colLetter = String.fromCharCode(64 + col);
      var head = headerVals[c] ? String(headerVals[c]) : '(col ' + col + ')';
      var val = values[r][c];
      var formula = formulas[r][c];
      var rtv = rich[r][c];

      log.push('  [' + colLetter + '] ' + head);
      log.push('    getValue()   = ' + JSON.stringify(val));
      if (formula) log.push('    getFormula() = ' + formula);
      if (rtv) {
        var text = rtv.getText();
        var cellLink = rtv.getLinkUrl();
        log.push('    rich.text    = ' + JSON.stringify(text));
        log.push('    rich.cellLink= ' + (cellLink || '(none)'));
        var runs = rtv.getRuns();
        log.push('    rich.runs[' + runs.length + ']:');
        for (var k = 0; k < runs.length; k++) {
          var run = runs[k];
          var s = run.getStartIndex();
          var e = run.getEndIndex();
          var chunk = text.substring(s, e);
          var ts = run.getTextStyle();
          var style = {
            bold: ts.isBold(),
            italic: ts.isItalic(),
            underline: ts.isUnderline(),
            strikethrough: ts.isStrikethrough(),
            fontFamily: ts.getFontFamily(),
            fontSize: ts.getFontSize(),
          };
          try {
            var color = ts.getForegroundColorObject && ts.getForegroundColorObject();
            if (color && color.asRgbColor) style.color = color.asRgbColor().asHexString();
          } catch (e2) { style.color = '(themed)'; }
          var runLink = run.getLinkUrl();
          log.push('      run[' + k + '] [' + s + ',' + e + ') ' + JSON.stringify(chunk));
          log.push('        style = ' + JSON.stringify(style));
          log.push('        link  = ' + (runLink || '(none)'));
        }
      } else {
        log.push('    rich.()      = null');
      }
    }
  }

  console.log(log.join('\n'));
  SpreadsheetApp.getUi().alert(
    'Dumped ' + log.length + ' lines. Open the Apps Script editor → View ▸ Executions → click the latest run.'
  );
}

/**
 * Dump each variable's chip URI (if any) and what extractDriveId / idMap
 * resolution would do, WITHOUT actually copying anything. Useful for
 * diagnosing why a chip in a copied doc didn't get re-pointed.
 */
function debugChipUris() {
  var ss = SpreadsheetApp.getActive();
  var vars = loadVariables(ss);
  var log = [];
  for (var i = 0; i < vars.length; i++) {
    var v = vars[i];
    if (!v.cellLinkUrl) continue;
    var id = extractDriveId(v.cellLinkUrl);
    log.push(v.name);
    log.push('  cellLinkUrl = ' + v.cellLinkUrl);
    log.push('  mimeType    = ' + v.mimeType);
    log.push('  extractedId = ' + (id || '(NO MATCH — extractDriveId could not parse)'));
  }
  console.log(log.join('\n'));
  SpreadsheetApp.getUi().alert('Dumped ' + log.length + ' lines. See execution log.');
}
