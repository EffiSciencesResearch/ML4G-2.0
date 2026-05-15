/**
 * For each (src, copy) pair, replay any directly-set (non-inherited) permissions
 * onto the copy. Inherited permissions are skipped — they come automatically
 * from the parent folder, which the copy shares with its source.
 */
function replayNonInheritedPermissions(copies) {
  for (var i = 0; i < copies.length; i++) {
    var c = copies[i];
    try {
      replayPermsForOne_(c.srcId, c.newId);
    } catch (e) {
      logError_('permissions', c.newId, e);
    }
  }
}

function replayPermsForOne_(srcId, newId) {
  var resp = withRetry_('Permissions.list', function () {
    return Drive.Permissions.list(srcId, {
      fields: 'permissions(id,type,role,emailAddress,domain,allowFileDiscovery,permissionDetails)',
      supportsAllDrives: true,
    });
  });
  var perms = resp.permissions || [];
  for (var i = 0; i < perms.length; i++) {
    var p = perms[i];
    if (p.role === 'owner') continue;
    // Skip permissions that are *only* inherited (no direct row).
    var details = p.permissionDetails || [];
    if (details.length > 0 && details.every(function (d) { return d.inherited === true; })) continue;

    var body = { type: p.type, role: p.role };
    if (p.type === 'user' || p.type === 'group') body.emailAddress = p.emailAddress;
    if (p.type === 'domain') body.domain = p.domain;
    if (p.allowFileDiscovery != null && (p.type === 'domain' || p.type === 'anyone')) {
      body.allowFileDiscovery = p.allowFileDiscovery;
    }
    try {
      withRetry_('Permissions.create', function () {
        return Drive.Permissions.create(body, newId, {
          supportsAllDrives: true,
          sendNotificationEmail: false,
        });
      });
    } catch (e) {
      logError_('permissions.create', newId, e);
    }
  }
}
