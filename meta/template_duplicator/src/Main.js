var CTRL_ROOT = '_FOLDER_OR_DOC_TO_DUPLICATE';

function runDuplicate() {
  var ss = SpreadsheetApp.getActive();
  var ui = SpreadsheetApp.getUi();

  toast_('Reading Template Variables…');
  var vars = loadVariables(ss);

  var rootVar = findVar(vars, CTRL_ROOT);
  if (!rootVar) {
    ui.alert('Missing row: ' + CTRL_ROOT + ' in the Template Variables table.');
    return;
  }
  var rootUrl = rootVar.cellLinkUrl || rootVar.plain;
  var rootSrcId = extractDriveId(rootUrl);
  if (!rootSrcId) {
    ui.alert(CTRL_ROOT + ' must be a Drive link (folder or file). Got: ' + rootUrl);
    return;
  }

  var substVars = vars.filter(function (v) { return v.name.charAt(0) !== '_'; });
  resolveNestedPlain(substVars);

  toast_('Copying tree…');
  // Place the new tree as a sibling of THIS spreadsheet (the duplicator).
  var targetParent = (Drive.Files.get(ss.getId(), { fields: 'parents', supportsAllDrives: true }).parents || [])[0];
  if (!targetParent) { ui.alert('Duplicator sheet has no parent folder — cannot place output.'); return; }
  var copyResult = copyTree(rootSrcId, substVars, targetParent);
  var idMap = copyResult.idMap;
  var copies = copyResult.copies;

  toast_('Rewriting in-tree URLs in values…');
  var formUrlCache = {};
  // Map srcId -> target file's mimeType, so rewriteUrl can detect forms
  // even when the value cell is a plain hyperlink (no chip).
  var srcMimeMap = {};
  for (var k = 0; k < copies.length; k++) srcMimeMap[copies[k].srcId] = copies[k].mimeType;
  for (var i = 0; i < substVars.length; i++) {
    var v = substVars[i];
    Object.keys(idMap).forEach(function (srcId) {
      if (v.plain.indexOf(srcId) !== -1) v.plain = v.plain.split(srcId).join(idMap[srcId]);
    });
    if (v.cellLinkUrl) {
      var srcId = extractDriveId(v.cellLinkUrl);
      var targetMime = (srcId && srcMimeMap[srcId]) || v.mimeType;
      v.cellLinkUrl = rewriteUrl(v.cellLinkUrl, idMap, formUrlCache, targetMime);
    }
  }
  resolveNestedPlain(substVars);

  toast_('Substituting in document bodies…');
  var warnings = substituteBodies(copies, substVars);
  logWarnings_(warnings);

  toast_('Replaying non-inherited permissions…');
  replayNonInheritedPermissions(copies);

  var rootNewId = copyResult.rootNewId;
  var rootFolder = Drive.Files.get(rootNewId, { fields: 'id,name,mimeType', supportsAllDrives: true });
  var rootLink = rootFolder.mimeType === MIME_FOLDER
    ? 'https://drive.google.com/drive/folders/' + rootNewId
    : 'https://drive.google.com/file/d/' + rootNewId;

  var msg = 'Done.\n\nNew root: ' + rootFolder.name + '\n' + rootLink;
  if (warnings.length) {
    msg += '\n\n⚠ Unknown placeholders found in ' + warnings.length + ' file(s):';
    for (var w = 0; w < warnings.length; w++) {
      msg += '\n  • ' + warnings[w].names.map(function (n) { return '{{' + n + '}}'; }).join(', ');
    }
    msg += '\n\nSee the "Run log" sheet for file links.';
  }
  ui.alert(msg);
}
