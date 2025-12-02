import os
import json
import time
import traceback
import gspread

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as UserCredentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload


class google_utils:

    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    def __init__(self, creds_path='credentials.json', token_path='token.json'):
        self.creds_path = creds_path
        self.token_path = token_path

        self.creds = None
        self.gc = None               # gspread client
        self.drive = None            # drive API

        self.authenticate()

    # =========================================================
    # AUTHENTICATION
    # =========================================================
    def authenticate(self):
        """Authenticate using either OAuth client or Service Account."""
        creds = None

        # Load user credentials (token)
        if os.path.exists(self.token_path):
            try:
                creds = UserCredentials.from_authorized_user_file(self.token_path, self.SCOPES)
            except:
                creds = None

        # Refresh or recreate creds
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception:
                    creds = None

            if not creds:
                if not os.path.exists(self.creds_path):
                    raise FileNotFoundError(f"Credentials file not found: {self.creds_path}")

                with open(self.creds_path, "r") as f:
                    data = json.load(f)

                # Service Account
                if data.get("type") == "service_account":
                    print("🔐 Using Service Account")
                    creds = ServiceAccountCredentials.from_service_account_file(
                        self.creds_path, scopes=self.SCOPES
                    )

                # OAuth User Login
                else:
                    print("🌐 Using OAuth Client — opening browser…")
                    flow = InstalledAppFlow.from_client_secrets_file(self.creds_path, self.SCOPES)
                    creds = flow.run_local_server(port=0)
                    with open(self.token_path, "w") as token:
                        token.write(creds.to_json())

        self.creds = creds
        self.gc = gspread.authorize(creds)
        self.drive = build('drive', 'v3', credentials=creds)

    # =========================================================
    # DRIVE HELPERS
    # =========================================================
    def resolve_folder_id(self, folder_id_or_name):
        """Return folder ID (search by name; fallback to ID)."""
        # Search by name
        query = (
            f"mimeType='application/vnd.google-apps.folder' "
            f"and name='{folder_id_or_name}' and trashed=false"
        )

        try:
            result = self.drive.files().list(q=query, fields="files(id,name)").execute()
            files = result.get("files", [])
            if files:
                return files[0]["id"]
        except HttpError:
            pass

        # Otherwise assume user gave folder ID
        return folder_id_or_name

    def find_file(self, name, folder_id):
        """Find Google Sheet by name inside folder."""
        query = (
            f"name='{name}' and '{folder_id}' in parents and "
            f"mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
        )

        try:
            result = self.drive.files().list(q=query, fields="files(id,name)").execute()
            files = result.get("files", [])
            return files[0]["id"] if files else None
        except HttpError:
            return None

    def create_sheet(self, name, folder_id):
        """Create a new Google Sheet in folder."""
        metadata = {
            "name": name,
            "parents": [folder_id],
            "mimeType": "application/vnd.google-apps.spreadsheet",
        }

        try:
            file = self.drive.files().create(body=metadata, fields="id").execute()
            return file.get("id")
        except HttpError:
            return None

    # =========================================================
    # SHEET WRITING
    # =========================================================
    def write_to_sheet(self, sheet_name, folder, data):
        """
        Create sheet if not exists, else append rows.
        """
        folder_id = self.resolve_folder_id(folder)

        # find existing sheet
        file_id = self.find_file(sheet_name, folder_id)

        # Write into existing sheet
        if file_id:
            sh = self.gc.open_by_key(file_id)
            ws = sh.sheet1
            ws.append_rows(data)
            print("✔ Data appended")
            return file_id

        # Create new sheet
        print("📄 Creating new sheet…")
        file_id = self.create_sheet(sheet_name, folder_id)
        if not file_id:
            raise Exception("Failed to create Google Sheet")

        time.sleep(3)  # wait for Google propagation

        sh = self.gc.open_by_key(file_id)
        ws = sh.sheet1
        ws.append_rows(data)

        print("✔ New sheet created and data added")
        return file_id

    # =========================================================
    # FILE UPLOAD
    # =========================================================
    def upload_file(self, file_path, folder, mime_type=None):
        """Upload ANY file to Google Drive."""
        folder_id = self.resolve_folder_id(folder)
        file_name = os.path.basename(file_path)

        metadata = {"name": file_name, "parents": [folder_id]}
        media = MediaFileUpload(file_path, mimetype=mime_type)

        file = self.drive.files().create(
            body=metadata, media_body=media, fields="id"
        ).execute()

        print(f"✔ Uploaded: {file_name}")
        return file.get("id")
