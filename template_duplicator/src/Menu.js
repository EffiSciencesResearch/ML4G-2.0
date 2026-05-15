function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('Template Duplicator')
    .addItem('Duplicate template…', 'runDuplicate')
    .addSeparator()
    .addItem('Debug: list tables', 'debugListTables')
    .addItem('Debug: dump variables', 'debugDumpVariables')
    .addItem('Debug: chip URIs', 'debugChipUris')
    .addToUi();
}
