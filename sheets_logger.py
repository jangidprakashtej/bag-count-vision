"""
Buffered logger that appends product-count events to a Google Sheet.

Requires a Google Cloud service account with the Sheets API enabled,
and the target spreadsheet shared with the service account's email
(found inside service_account.json as "client_email"). See README.md.
"""
import time
import datetime
import threading
import gspread
from google.oauth2.service_account import Credentials

import config

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]


class SheetsLogger:
    def __init__(self):
        creds = Credentials.from_service_account_file(
            config.GOOGLE_SERVICE_ACCOUNT_JSON, scopes=SCOPES
        )
        client = gspread.authorize(creds)
        sheet = client.open(config.GOOGLE_SHEET_NAME)
        self.worksheet = sheet.worksheet(config.GOOGLE_WORKSHEET_NAME)

        self._buffer = []
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()

        self._ensure_header()

    def _ensure_header(self):
        first_row = self.worksheet.row_values(1)
        expected = ["Timestamp", "Product", "Track ID", "Running Count"]
        if first_row != expected:
            self.worksheet.insert_row(expected, index=1)

    def log_event(self, product, track_id, running_count):
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        with self._lock:
            self._buffer.append([timestamp, product, track_id, running_count])

    def _flush_loop(self):
        while self._running:
            time.sleep(config.SHEETS_FLUSH_INTERVAL_SEC)
            self.flush()

    def flush(self):
        with self._lock:
            if not self._buffer:
                return
            rows, self._buffer = self._buffer, []
        try:
            self.worksheet.append_rows(rows, value_input_option="USER_ENTERED")
        except Exception as e:
            print(f"[ERROR] failed to write to Google Sheet: {e}")
            # put rows back so we retry on the next flush
            with self._lock:
                self._buffer = rows + self._buffer

    def stop(self):
        self._running = False
        self.flush()
