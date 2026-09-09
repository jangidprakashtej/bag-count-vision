"""
Buffered logger that appends product-count events to a Google Sheet.

Refactored to take the app's settings dict (from settings_manager.py)
instead of the old config.py module, so it works with the GUI app.

Requires a Google Cloud service account with the Sheets API AND Drive
API enabled, and the target spreadsheet shared with the service
account's email (found inside service_account.json as "client_email").
"""
import time
import datetime
import threading
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

HEADER = ["Timestamp", "Factory", "Product", "Track ID", "Running Count"]


class SheetsLogger:
    def __init__(self, settings, on_error=None):
        self.settings = settings
        self.factory_name = settings.get("factory_name", "Factory")
        self.on_error = on_error  # optional callback(str) for surfacing failures to a GUI
        self._consecutive_failures = 0

        self._connect()

        self._buffer = []
        self._lock = threading.Lock()
        self._running = True
        self._flush_interval = settings.get("sheets_flush_interval_sec", 5)
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()

        self._ensure_header()

    def _connect(self):
        creds = Credentials.from_service_account_file(
            self.settings["service_account_json"], scopes=SCOPES
        )
        client = gspread.authorize(creds)
        sheet = client.open(self.settings["google_sheet_name"])
        self.worksheet = sheet.worksheet(self.settings["google_worksheet_name"])

    def _ensure_header(self):
        first_row = self.worksheet.row_values(1)
        if first_row != HEADER:
            self.worksheet.insert_row(HEADER, index=1)

    def log_event(self, product, track_id, running_count):
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        with self._lock:
            self._buffer.append([timestamp, self.factory_name, product, track_id, running_count])

    def get_todays_counts(self):
        """
        Reads existing rows already in the Sheet and returns a
        {category: count} dict for TODAY, for THIS factory only. Called
        at startup so a restarted app (e.g. after a power outage)
        resumes counting from where it left off instead of starting
        back at zero. A new calendar day naturally starts fresh at 1.
        """
        try:
            rows = self.worksheet.get_all_records(expected_headers=HEADER)
        except Exception as e:
            print(f"[WARN] could not load existing counts to resume from: {e}")
            return {}

        today = datetime.date.today().isoformat()
        counts = {}
        for row in rows:
            ts = str(row.get("Timestamp", ""))
            factory = row.get("Factory", "")
            if ts.startswith(today) and factory == self.factory_name:
                category = row.get("Product", "UNKNOWN")
                counts[category] = counts.get(category, 0) + 1
        return counts

    def _flush_loop(self):
        while self._running:
            time.sleep(self._flush_interval)
            self.flush()

    def flush(self):
        with self._lock:
            if not self._buffer:
                return
            rows, self._buffer = self._buffer, []
        try:
            self.worksheet.append_rows(rows, value_input_option="USER_ENTERED")
            if self._consecutive_failures > 0:
                # we were failing, now recovered -- let the user know
                recovered_msg = "Google Sheets: connection recovered, writes resuming normally."
                print(recovered_msg)
                if self.on_error:
                    self.on_error(recovered_msg)
            self._consecutive_failures = 0
        except Exception as e:
            self._consecutive_failures += 1
            message = f"[ERROR] failed to write to Google Sheet (attempt {self._consecutive_failures}): {e}"
            print(message)

            # Only bother the user after several consecutive failures --
            # a single transient network hiccup shouldn't pop up an alert,
            # but a sustained problem should.
            if self._consecutive_failures >= 3 and self.on_error:
                self.on_error(
                    f"Google Sheets: {self._consecutive_failures} failed write attempts in a row. "
                    f"Retrying automatically -- data is buffered locally, not lost."
                )

            # After a longer stretch of failures, the connection itself may be
            # stale (e.g. network dropped and came back) -- try to reconnect
            # from scratch rather than endlessly retrying a dead connection.
            if self._consecutive_failures % 6 == 0:
                try:
                    print("Attempting to reconnect to Google Sheets...")
                    self._connect()
                except Exception as reconnect_error:
                    print(f"[WARN] reconnect attempt failed: {reconnect_error}")

            with self._lock:
                self._buffer = rows + self._buffer

    def stop(self):
        self._running = False
        self.flush()
