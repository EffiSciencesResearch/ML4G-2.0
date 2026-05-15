/**
 * Substitute variables in copied Docs/Slides.
 *
 * Each variable is treated as plain text + optional link URL (`cellLinkUrl`,
 * which may come from a hyperlink applied to the cell or from a Sheets smart
 * chip). A chip in a Doc/Slide is rendered as a clickable hyperlink (the Docs
 * API does not support inserting real smart chips programmatically).
 *
 * Returns a list of warnings { fileUrl, names: [...] } for placeholders that
 * didn't map to any variable.
 */
function substituteBodies(copies, vars) {
  var warnings = [];
  for (var i = 0; i < copies.length; i++) {
    var c = copies[i];
    try {
      if (c.mimeType === MIME_DOC) {
        var unknownDoc = substituteDoc_(c.newId, vars);
        if (unknownDoc.length) warnings.push({ fileUrl: 'https://docs.google.com/document/d/' + c.newId, names: unknownDoc });
      } else if (c.mimeType === MIME_SLIDES) {
        var unknownSlides = substituteSlides_(c.newId, vars);
        if (unknownSlides.length) warnings.push({ fileUrl: 'https://docs.google.com/presentation/d/' + c.newId, names: unknownSlides });
      } else if (c.mimeType === MIME_FORM) {
        var unknownForm = substituteForm_(c.newId, vars);
        if (unknownForm.length) warnings.push({ fileUrl: 'https://docs.google.com/forms/d/' + c.newId + '/edit', names: unknownForm });
      }
    } catch (e) {
      logError_('substitute', c.newId, e);
    }
  }
  return warnings;
}

/* ---------------- Google Docs ---------------- */

function substituteDoc_(docId, vars) {
  // Pass 1 — chip variables (those whose source cell was a Sheets smart chip).
  // We insert real Docs smart chips via the Docs REST API BEFORE opening with
  // DocumentApp (which would cache the doc and miss our edits).
  // Forms are excluded: their respondent URL isn't a Drive file URI, so
  // insertRichLink wouldn't produce a real chip. They go through the
  // DocumentApp hyperlink path below instead.
  var chipVars = vars.filter(function (v) {
    return v.name.charAt(0) !== '_' && v.mimeType && v.mimeType !== MIME_FORM && v.cellLinkUrl;
  });
  if (chipVars.length) {
    try { insertChipsInDoc(docId, chipVars); }
    catch (e) { logError_('chip-pass', docId, e); }
  }

  // Pass 2 — text / hyperlink variables, via DocumentApp.
  var doc = DocumentApp.openById(docId);
  var sections = [doc.getBody()];
  var header = doc.getHeader(); if (header) sections.push(header);
  var footer = doc.getFooter(); if (footer) sections.push(footer);

  for (var s = 0; s < sections.length; s++) {
    var section = sections[s];
    for (var i = 0; i < vars.length; i++) {
      var v = vars[i];
      if (v.name.charAt(0) === '_') continue;
      if (v.mimeType && v.mimeType !== MIME_FORM) continue; // already handled by chip pass
      if (v.cellLinkUrl) {
        replaceWithLinkInDoc_(section, v);
      } else {
        section.replaceText('\\{\\{' + escapeForRegex_(v.name) + '\\}\\}', v.plain);
      }
    }
  }
  // Collect leftover {{...}} text before saving (cheap: we already have the doc open).
  var leftover = '';
  for (var s2 = 0; s2 < sections.length; s2++) leftover += '\n' + sections[s2].getText();
  doc.saveAndClose();
  return scanUnknownPlaceholders_(leftover, vars);
}

/**
 * Replace every {{NAME}} in `section` with v.plain, then mark the inserted
 * range as a hyperlink to v.cellLinkUrl.
 */
function replaceWithLinkInDoc_(section, v) {
  var pattern = '\\{\\{' + escapeForRegex_(v.name) + '\\}\\}';
  var safety = 0;
  var range = section.findText(pattern);
  while (range && safety++ < 1000) {
    var el = range.getElement();
    if (!el || el.editAsText == null) {
      range = section.findText(pattern, range);
      continue;
    }
    var text = el.editAsText();
    var start = range.getStartOffset();
    var end = range.getEndOffsetInclusive();

    text.deleteText(start, end);
    if (v.plain.length === 0) {
      range = section.findText(pattern);
      continue;
    }
    text.insertText(start, v.plain);
    var from = start;
    var to = start + v.plain.length - 1;
    try { text.setLinkUrl(from, to, v.cellLinkUrl); } catch (e) {}
    range = section.findText(pattern);
  }
}

/* ---------------- Google Slides ---------------- */

function substituteSlides_(presId, vars) {
  var pres = SlidesApp.openById(presId);

  // Plain (no-link) vars first via the built-in fast path.
  for (var i = 0; i < vars.length; i++) {
    var v = vars[i];
    if (v.name.charAt(0) === '_') continue;
    if (v.cellLinkUrl) continue;
    pres.replaceAllText('{{' + v.name + '}}', v.plain);
  }

  // Linked vars: locate, replace, apply link.
  var slides = pres.getSlides();
  for (var s = 0; s < slides.length; s++) {
    var ranges = collectSlideTextRanges_(slides[s]);
    for (var r = 0; r < ranges.length; r++) {
      var tr = ranges[r];
      for (var j = 0; j < vars.length; j++) {
        var lv = vars[j];
        if (lv.name.charAt(0) === '_') continue;
        if (!lv.cellLinkUrl) continue;
        replaceWithLinkInSlideText_(tr, lv);
      }
    }
  }
  // Collect leftover {{...}} text before saving.
  var leftover = '';
  slides = pres.getSlides();
  for (var s3 = 0; s3 < slides.length; s3++) {
    var ranges3 = collectSlideTextRanges_(slides[s3]);
    for (var r3 = 0; r3 < ranges3.length; r3++) leftover += '\n' + ranges3[r3].asString();
  }
  pres.saveAndClose();
  return scanUnknownPlaceholders_(leftover, vars);
}

function replaceWithLinkInSlideText_(textRange, v) {
  var needle = '{{' + v.name + '}}';
  var needlePattern = escapeForRegex_(needle);
  var plainPattern = escapeForRegex_(v.plain);
  var safety = 0;
  while (safety++ < 1000) {
    var matches = textRange.find(needlePattern);
    if (!matches || !matches.length) return;
    var m = matches[0];
    var replaced = m.getRange().replaceAllText(needle, v.plain);
    if (!replaced) return;
    if (v.plain.length > 0) {
      var inserted = textRange.find(plainPattern);
      if (inserted && inserted.length) {
        try { inserted[0].getTextStyle().setLinkUrl(v.cellLinkUrl); } catch (e) {}
      }
    }
  }
}

function collectSlideTextRanges_(slide) {
  var out = [];
  var shapes = slide.getShapes();
  for (var i = 0; i < shapes.length; i++) {
    var sh = shapes[i];
    if (sh.getText) out.push(sh.getText());
  }
  var tables = slide.getTables();
  for (var t = 0; t < tables.length; t++) {
    var tbl = tables[t];
    for (var ri = 0; ri < tbl.getNumRows(); ri++) {
      for (var ci = 0; ci < tbl.getNumColumns(); ci++) {
        out.push(tbl.getCell(ri, ci).getText());
      }
    }
  }
  var groups = slide.getGroups();
  for (var g = 0; g < groups.length; g++) {
    var grpShapes = groups[g].getChildren();
    for (var k = 0; k < grpShapes.length; k++) {
      if (grpShapes[k].getText) out.push(grpShapes[k].getText());
    }
  }
  return out;
}

/* ---------------- Google Forms ---------------- */

function substituteForm_(formId, vars) {
  var form = FormApp.openById(formId);
  var leftover = '';

  function subst(s) {
    if (s == null) return s;
    var out = String(s);
    for (var i = 0; i < vars.length; i++) {
      var v = vars[i];
      if (v.name.charAt(0) === '_') continue;
      out = out.split('{{' + v.name + '}}').join(v.plain);
    }
    leftover += '\n' + out;
    return out;
  }

  form.setTitle(subst(form.getTitle()));
  form.setDescription(subst(form.getDescription()));
  try { form.setConfirmationMessage(subst(form.getConfirmationMessage())); } catch (e) {}

  var items = form.getItems();
  for (var i = 0; i < items.length; i++) {
    var item = items[i];
    try { item.setTitle(subst(item.getTitle())); } catch (e) {}
    try { item.setHelpText(subst(item.getHelpText())); } catch (e) {}
    // Multiple-choice / checkbox / list items: substitute each choice value.
    var type = item.getType();
    var typed = null;
    if (type === FormApp.ItemType.MULTIPLE_CHOICE) typed = item.asMultipleChoiceItem();
    else if (type === FormApp.ItemType.CHECKBOX) typed = item.asCheckboxItem();
    else if (type === FormApp.ItemType.LIST) typed = item.asListItem();
    if (typed) {
      var choices = typed.getChoices();
      var newChoices = [];
      for (var c = 0; c < choices.length; c++) {
        newChoices.push(typed.createChoice(subst(choices[c].getValue())));
      }
      try { typed.setChoices(newChoices); } catch (e) {}
    }
  }

  return scanUnknownPlaceholders_(leftover, vars);
}

/* ---------------- Shared helpers ---------------- */

function scanUnknownPlaceholders_(text, vars) {
  var known = {};
  for (var i = 0; i < vars.length; i++) known[vars[i].name] = true;
  var seen = {};
  var re = /\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}/g;
  var m;
  while ((m = re.exec(text)) !== null) {
    var n = m[1];
    if (!known[n] && n.charAt(0) !== '_') seen[n] = true;
  }
  return Object.keys(seen);
}

function escapeForRegex_(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
