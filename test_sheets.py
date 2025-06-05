from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import os

# 1. Path to your service account JSON
creds = Credentials.from_service_account_file(
    os.path.join("credentials", "service_account_key.json"),
    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"]
)

# 2. Build the Sheets service
service = build("sheets", "v4", credentials=creds)
sheet = service.spreadsheets()

# 3. Read the first row (A1:E1) of your sheet to verify access
SPREADSHEET_ID = os.getenv("GOOGLE_SHEET_ID", "YOUR_SHEET_ID_HERE")
RANGE = "Sheet1!A1:E1"  # adjust if your tab is named differently

result = sheet.values().get(
    spreadsheetId=SPREADSHEET_ID,
    range=RANGE
).execute()
values = result.get("values", [])

if not values:
    print("No data found.")
else:
    print("Row 1 (headers) in the sheet:")
    print(values[0])
