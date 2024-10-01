# How to use scripts to automate tasks with Google Drive/Docs.


**Duplicate Career Docs**

Before running the script, you need to have: 

- a Google Drive folder where you want to store the documents.
- a Google Docs template that you want to use as a base for the new documents.
- a CSV `names_and_emails.csv` file with the names and emails of the people you want to create documents for. (column names: )
- a Google Drive API `service_account_token.json` that you can get from the Google Cloud Console. You need to give the service account write access to the folder where you want to store the documents and view access to the template.

Run the script with the following command:

```shell
python googletools.py duplicate-career-docs names_and_emails.csv template_url drive_folder_url
```

Diego will write the readme relative to the other scripts.
