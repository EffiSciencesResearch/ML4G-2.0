var MIME_FOLDER = 'application/vnd.google-apps.folder';
var MIME_DOC = 'application/vnd.google-apps.document';
var MIME_SLIDES = 'application/vnd.google-apps.presentation';
var MIME_SHORTCUT = 'application/vnd.google-apps.shortcut';

/**
 * Recursively copies the source tree as a sibling of the source.
 * Returns: { rootNewId, idMap (srcId -> newId), copies (array of {srcId, newId, mimeType}) }.
 *
 * Title substitution uses resolved plain values from `vars`.
 */
function copyTree(srcRootId, vars, targetParent) {
  var srcRoot = withRetry_('Files.get(root)', function () {
    return Drive.Files.get(srcRootId, { fields: 'id,name,mimeType,parents', supportsAllDrives: true });
  });
  var siblingParent = targetParent || (srcRoot.parents && srcRoot.parents[0]) || null;
  if (!siblingParent) {
    throw new Error('No parent folder available to place the copy.');
  }

  var idMap = {};
  var copies = [];
  var pendingShortcuts = []; // { name, parentNewId, targetId }

  var newRootId;
  var rootName = substituteTitle(srcRoot.name, vars);
  if (srcRoot.mimeType === MIME_FOLDER) {
    newRootId = createFolder(rootName, siblingParent);
    idMap[srcRoot.id] = newRootId;
    copies.push({ srcId: srcRoot.id, newId: newRootId, mimeType: MIME_FOLDER });
    copyFolderContents_(srcRoot.id, newRootId, vars, idMap, copies, pendingShortcuts);
  } else {
    var copied = withRetry_('Files.copy(root)', function () {
      return Drive.Files.copy({ name: rootName, parents: [siblingParent] }, srcRoot.id, { supportsAllDrives: true });
    });
    newRootId = copied.id;
    idMap[srcRoot.id] = newRootId;
    copies.push({ srcId: srcRoot.id, newId: newRootId, mimeType: srcRoot.mimeType });
  }

  // Create shortcuts last — by now idMap has all in-tree targets resolved.
  for (var si = 0; si < pendingShortcuts.length; si++) {
    var sc = pendingShortcuts[si];
    var resolvedTarget = idMap[sc.targetId] || sc.targetId;
    try {
      withRetry_('Files.create(shortcut)', function () {
        return Drive.Files.create({
          name: sc.name,
          mimeType: MIME_SHORTCUT,
          parents: [sc.parentNewId],
          shortcutDetails: { targetId: resolvedTarget },
        }, null, { supportsAllDrives: true });
      });
    } catch (e) {
      logError_('shortcut-create', sc.targetId, e);
    }
  }

  return { rootNewId: newRootId, idMap: idMap, copies: copies };
}

function copyFolderContents_(srcFolderId, dstFolderId, vars, idMap, copies, pendingShortcuts) {
  var pageToken = null;
  do {
    var resp = withRetry_('Files.list', function () {
      return Drive.Files.list({
        q: "'" + srcFolderId + "' in parents and trashed = false",
        fields: 'nextPageToken, files(id,name,mimeType,shortcutDetails(targetId))',
        pageSize: 1000,
        supportsAllDrives: true,
        includeItemsFromAllDrives: true,
        pageToken: pageToken,
      });
    });
    var files = resp.files || [];
    for (var i = 0; i < files.length; i++) {
      var f = files[i];
      var newName = substituteTitle(f.name, vars);
      if (f.mimeType === MIME_FOLDER) {
        var newFolderId = createFolder(newName, dstFolderId);
        idMap[f.id] = newFolderId;
        copies.push({ srcId: f.id, newId: newFolderId, mimeType: MIME_FOLDER });
        copyFolderContents_(f.id, newFolderId, vars, idMap, copies, pendingShortcuts);
      } else if (f.mimeType === MIME_SHORTCUT) {
        // Defer: in-tree target may not be copied yet.
        pendingShortcuts.push({
          name: newName,
          parentNewId: dstFolderId,
          targetId: f.shortcutDetails && f.shortcutDetails.targetId,
        });
      } else {
        var copied = withRetry_('Files.copy(' + f.name + ')', function () {
          return Drive.Files.copy({ name: newName, parents: [dstFolderId] }, f.id, { supportsAllDrives: true });
        });
        idMap[f.id] = copied.id;
        copies.push({ srcId: f.id, newId: copied.id, mimeType: f.mimeType });
      }
    }
    pageToken = resp.nextPageToken;
  } while (pageToken);
}

function createFolder(name, parentId) {
  var folder = withRetry_('Files.create(folder)', function () {
    return Drive.Files.create({
      name: name,
      mimeType: MIME_FOLDER,
      parents: [parentId],
    }, null, { supportsAllDrives: true });
  });
  return folder.id;
}

/** Drive-file-ID extractor for URLs that look like Drive/Docs/Slides/Sheets/Forms links. */
function extractDriveId(url) {
  if (!url) return null;
  var patterns = [
    /\/(?:document|presentation|spreadsheets|forms)\/(?:u\/\d+\/)?d\/([a-zA-Z0-9_-]{20,})/,
    /\/folders\/([a-zA-Z0-9_-]{20,})/,
    /\/file\/d\/([a-zA-Z0-9_-]{20,})/,
    /[?&]id=([a-zA-Z0-9_-]{20,})/,
  ];
  for (var i = 0; i < patterns.length; i++) {
    var m = url.match(patterns[i]);
    if (m) return m[1];
  }
  return null;
}

var MIME_FORM = 'application/vnd.google-apps.form';

/**
 * Replace any in-tree Drive ID embedded in `url` with the new copy's ID.
 * Caller may pass `mimeType` so we can special-case forms (whose respondent
 * URL uses a separate published-id and isn't reachable by naive ID swap).
 */
function rewriteUrl(url, idMap, formUrlCache, mimeType) {
  if (!url) return url;
  var srcId = extractDriveId(url);
  if (!srcId || !idMap[srcId]) return url;
  var newId = idMap[srcId];

  if (mimeType === MIME_FORM) {
    if (formUrlCache && formUrlCache[newId]) return formUrlCache[newId];
    try {
      var form = FormApp.openById(newId);
      try { form.setPublished(true); } catch (e1) { logError_('form-setPublished', newId, e1); }
      try { form.setAcceptingResponses(true); } catch (e2) { logError_('form-setAcceptingResponses', newId, e2); }
      var pub = form.getPublishedUrl();
      if (formUrlCache) formUrlCache[newId] = pub;
      return pub;
    } catch (e) {
      logError_('form-published-url', newId, e);
    }
  }
  return url.replace(srcId, newId);
}
