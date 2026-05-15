var RUN_LOG_SHEET = 'Run log';

function toast_(msg) {
  try { SpreadsheetApp.getActive().toast(msg, 'Duplicator', 5); } catch (e) {}
}

function logError_(phase, id, err) {
  var ss = SpreadsheetApp.getActive();
  var sheet = ss.getSheetByName(RUN_LOG_SHEET) || ss.insertSheet(RUN_LOG_SHEET);
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(['Time', 'Phase', 'File ID', 'Message']);
  }
  var msg = err && err.message ? err.message : String(err);
  sheet.appendRow([new Date(), phase, id, msg]);
  console.error(phase + ' / ' + id + ': ' + msg + (err && err.stack ? '\n' + err.stack : ''));
}

/**
 * Run `fn` with exponential-backoff retry on transient Drive errors
 * (userRateLimitExceeded, rateLimitExceeded, backendError, 429, 5xx).
 * Sleeps 1s, 2s, 4s, 8s, 16s with ±25% jitter; gives up after 5 retries.
 */
function withRetry_(label, fn) {
  var delays = [1000, 2000, 4000, 8000, 16000];
  for (var attempt = 0; attempt <= delays.length; attempt++) {
    try {
      return fn();
    } catch (e) {
      var msg = (e && e.message) || String(e);
      var retryable = /rate ?limit|userRateLimit|backendError|quotaExceeded|\b429\b|\b50[023]\b|internal error/i.test(msg);
      if (!retryable || attempt === delays.length) throw e;
      var base = delays[attempt];
      var jittered = base + Math.floor((Math.random() - 0.5) * base * 0.5);
      console.log(label + ': retry ' + (attempt + 1) + ' after ' + jittered + 'ms — ' + msg);
      Utilities.sleep(jittered);
    }
  }
}

function logWarnings_(warnings) {
  if (!warnings || !warnings.length) return;
  var ss = SpreadsheetApp.getActive();
  var sheet = ss.getSheetByName(RUN_LOG_SHEET) || ss.insertSheet(RUN_LOG_SHEET);
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(['Time', 'Phase', 'File ID', 'Message']);
  }
  for (var i = 0; i < warnings.length; i++) {
    var w = warnings[i];
    sheet.appendRow([new Date(), 'unknown-placeholder', w.fileUrl, w.names.join(', ')]);
  }
}
