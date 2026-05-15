/**
 * Insert real Google Docs smart chips (RichLink elements) for variables whose
 * source cell contained a Sheets smart chip (i.e. v.mimeType is set).
 *
 * For each occurrence of {{VAR}} in the body, we issue a batchUpdate of:
 *   - deleteContentRange  to remove the placeholder text
 *   - insertRichLink      to put a chip at the placeholder's start index
 *
 * Requests must be processed back-to-front (descending startIndex) so earlier
 * indices remain valid after each delete/insert.
 *
 * Note: DocumentApp's findText returns *element-local* offsets, but the Docs
 * REST API needs *document-absolute* indices. We compute those by walking the
 * document structure ourselves.
 */
function insertChipsInDoc(docId, chipVars) {
  if (!chipVars.length) return;
  var doc = Docs.Documents.get(docId);
  var occurrences = [];
  collectPlaceholderOccurrences_(doc.body, chipVars, occurrences);
  if (!occurrences.length) return;

  // Process descending so earlier indices stay valid; bundle into one batchUpdate.
  occurrences.sort(function (a, b) { return b.startIndex - a.startIndex; });
  var requests = [];
  for (var i = 0; i < occurrences.length; i++) {
    var occ = occurrences[i];
    requests.push({ deleteContentRange: { range: { startIndex: occ.startIndex, endIndex: occ.endIndex } } });
    requests.push({ insertRichLink: { richLinkProperties: { uri: occ.uri }, location: { index: occ.startIndex } } });
  }
  try {
    Docs.Documents.batchUpdate({ requests: requests }, docId);
  } catch (e) {
    logError_('chip-insert', docId, e);
  }
}

/**
 * Walks the document body and records each placeholder occurrence as
 * { startIndex, endIndex, uri } in `out`.
 */
function collectPlaceholderOccurrences_(body, chipVars, out) {
  if (!body || !body.content) return;
  for (var i = 0; i < body.content.length; i++) {
    var struct = body.content[i];
    if (struct.paragraph) {
      walkParagraph_(struct.paragraph, chipVars, out);
    } else if (struct.table) {
      var table = struct.table;
      for (var r = 0; r < (table.tableRows || []).length; r++) {
        var row = table.tableRows[r];
        for (var c = 0; c < (row.tableCells || []).length; c++) {
          collectPlaceholderOccurrences_(row.tableCells[c], chipVars, out);
        }
      }
    }
  }
}

function walkParagraph_(paragraph, chipVars, out) {
  // Concatenate all textRuns in the paragraph and remember each char's absolute
  // index, so we find placeholders even when Docs has split them across runs
  // (e.g. inside table cells where styling boundaries fragment the text).
  var elements = paragraph.elements || [];
  var concat = '';
  var absIndex = []; // absIndex[i] = absolute doc index of concat.charAt(i)
  for (var i = 0; i < elements.length; i++) {
    var el = elements[i];
    if (!el.textRun) continue;
    var content = el.textRun.content || '';
    var elStart = el.startIndex;
    for (var k = 0; k < content.length; k++) {
      concat += content.charAt(k);
      absIndex.push(elStart + k);
    }
  }
  for (var v = 0; v < chipVars.length; v++) {
    var cv = chipVars[v];
    var needle = '{{' + cv.name + '}}';
    var idx = 0;
    while ((idx = concat.indexOf(needle, idx)) !== -1) {
      out.push({
        startIndex: absIndex[idx],
        endIndex: absIndex[idx + needle.length - 1] + 1,
        uri: cv.cellLinkUrl,
      });
      idx += needle.length;
    }
  }
}
