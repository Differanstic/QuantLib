import os
import json
import gspread
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as UserCredentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

def authenticate_google(creds_path='credentials.json', token_path='token.json'):
    """
    Authenticates with Google Sheets and Drive API using either Service Account or OAuth Client ID.
    """
    creds = None
    
    # 1. Try loading existing user credentials (token.json)
    if os.path.exists(token_path):
        try:
            creds = UserCredentials.from_authorized_user_file(token_path, SCOPES)
        except Exception:
            creds = None

    # 2. If no valid user creds, try service account or new login
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds:
            if not os.path.exists(creds_path):
                print(f"Error: Credentials file '{creds_path}' not found.")
                return None, None

            # Determine credential type
            try:
                with open(creds_path) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading credentials file: {e}")
                return None, None

            if 'type' in data and data['type'] == 'service_account':
                # Service Account
                print("Using Service Account credentials...")
                creds = ServiceAccountCredentials.from_service_account_file(creds_path, scopes=SCOPES)
            elif 'installed' in data or 'web' in data:
                # OAuth Client ID
                print("Using OAuth Client ID. Launching browser for authentication...")
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                creds = flow.run_local_server(port=0)
                # Save the credentials for the next run
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            else:
                print("Unknown credential type in credentials.json. Expected Service Account or OAuth Client ID.")
                return None, None

    try:
        gc = gspread.authorize(creds)
        drive_service = build('drive', 'v3', credentials=creds)
        return gc, drive_service
    except Exception as e:
        print(f"Error authorizing clients: {e}")
        return None, None

def resolve_folder_id(drive_service, folder_id_or_name):
    """
    Tries to find a folder by name. If found, returns its ID.
    If not found, assumes the input is already an ID and returns it.
    """
    try:
        # Search for folder by name
        query = f"mimeType = 'application/vnd.google-apps.folder' and name = '{folder_id_or_name}' and trashed = false"
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        if files:
            print(f"Found folder '{folder_id_or_name}' with ID: {files[0]['id']}")
            return files[0]['id']
        
        # If not found by name, assume it's an ID
        # Optional: Verify if it exists as an ID, but we can just return it.
        return folder_id_or_name
    except HttpError as error:
        print(f"Warning: Error resolving folder ID: {error}")
        return folder_id_or_name

def find_file_in_folder(drive_service, file_name, folder_id):
    """
    Searches for a file with the given name in the specified folder.
    Returns the file ID if found, else None.
    """
    try:
        query = f"name = '{file_name}' and '{folder_id}' in parents and mimeType = 'application/vnd.google-apps.spreadsheet' and trashed = false"
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        if files:
            return files[0]['id']
        return None
    except HttpError as error:
        print(f"An error occurred searching for file: {error}")
        return None

def create_sheet_in_folder(drive_service, file_name, folder_id):
    """
    Creates a new Google Sheet in the specified folder.
    """
    try:
        file_metadata = {
            'name': file_name,
            'parents': [folder_id],
            'mimeType': 'application/vnd.google-apps.spreadsheet'
        }
        file = drive_service.files().create(body=file_metadata, fields='id').execute()
        return file.get('id')
    except HttpError as error:
        print(f"An error occurred creating file: {error}")
        return None

def write_to_sheet(file_name, folder_id_or_name, data, creds_path='credentials.json'):
    """
    Writes data to a Google Sheet or appends data if it already exists in the folder.

    Usage:
        data = [['Name', 'Age'], ['Alice', 30]]
        write_to_sheet('MySheet', 'DB', data)
    """
    import time
    import traceback

    gc, drive_service = authenticate_google(creds_path)
    if not gc or not drive_service:
        return

    # Resolve folder ID (handle name vs ID)
    folder_id = resolve_folder_id(drive_service, folder_id_or_name)

    # Check if file exists
    file_id = find_file_in_folder(drive_service, file_name, folder_id)

    if file_id:
        print(f"File '{file_name}' found (ID: {file_id}). Appending data...")
        try:
            sh = gc.open_by_key(file_id)
            worksheet = sh.sheet1
            worksheet.append_rows(data)
            print("File Saved!")
        except Exception as e:
            print(f"Error appending data: {e}")
            traceback.print_exc()
    else:
        print(f"File '{file_name}' not found. Creating new sheet...")
        file_id = create_sheet_in_folder(drive_service, file_name, folder_id)
        if file_id:
            print(f"File created with ID: {file_id}. Waiting for propagation...")
            time.sleep(5) 
            try:
                sh = gc.open_by_key(file_id)
                worksheet = sh.sheet1
                worksheet.append_rows(data)
                print(f"File Saved!")
            except Exception as e:
                print(f"Error writing to new sheet: {repr(e)}")
                traceback.print_exc()
        else:
            print("Failed to create new sheet.")

def upload_file(file_path, folder_id_or_name, mime_type=None, creds_path='credentials.json'):
    """
    Uploads a file to Google Drive.

    Usage:
        upload_file('report.pdf', 'Reports', mime_type='application/pdf')
    """
    gc, drive_service = authenticate_google(creds_path)
    if not gc or not drive_service:
        return

    filename = os.path.basename(file_path)
    folder_id = resolve_folder_id(drive_service, folder_id_or_name)

    file_metadata = {'name': filename}
    if folder_id:
        file_metadata['parents'] = [folder_id]

    try:
        media = MediaFileUpload(file_path, mimetype=mime_type)
        file = drive_service.files().create(body=file_metadata,
                                        media_body=media,
                                        fields='id').execute()
        print(f"✔ File uploaded successfully. File ID: {file.get('id')}")
        return file.get('id')
    except Exception as e:
        print(f"❌ An error occurred uploading file: {e}")
        return None

if __name__ == "__main__":
    # Example Usage
    FOLDER_ID = 'FOLDER_NAME' # Can be name or ID now
    CREDS_FILE = 'credentials.json'
    FILE_NAME = 'Test_Sheet'
    DATA = [
        ['Name', 'Age', 'City'],
        ['Alice', 30, 'New York'],
        ['Bob', 25, 'Los Angeles']
    ]
    
    # write_to_sheet(FILE_NAME, FOLDER_ID, DATA, CREDS_FILE)

