"""
Daily Report — summarizes today's (or a given date's) bag counts by
category into tonnage, and appends a row per category to a
"Daily Report" worksheet in the same Google Sheet.

There's no need to literally "reset" the running counter in main.py --
every logged row already has its own timestamp, so a "reset" is just
"only look at rows from today." This script does that filtering.

Run manually:
    python daily_report.py                  # summarizes today
    python daily_report.py 2026-07-12        # summarizes a specific date

Or schedule it to run automatically every day at midnight, e.g. with cron
(crontab -e), adding a line like:
    1 0 * * * cd /path/to/project && /path/to/python3.11 daily_report.py
"""
import sys
import datetime
from collections import defaultdict

import gspread
from google.oauth2.service_account import Credentials

import config

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]


def main():
    target_date = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().isoformat()

    creds = Credentials.from_service_account_file(config.GOOGLE_SERVICE_ACCOUNT_JSON, scopes=SCOPES)
    client = gspread.authorize(creds)
    sheet = client.open(config.GOOGLE_SHEET_NAME)
    log_ws = sheet.worksheet(config.GOOGLE_WORKSHEET_NAME)

    rows = log_ws.get_all_records()  # each row: Timestamp, Product, Track ID, Running Count

    counts = defaultdict(int)
    for row in rows:
        ts = str(row.get("Timestamp", ""))
        if ts.startswith(target_date):
            counts[row.get("Product", "UNKNOWN")] += 1

    if not counts:
        print(f"No log entries found for {target_date}.")
        return

    # Prepare the Daily Report worksheet (create it if it doesn't exist yet)
    try:
        report_ws = sheet.worksheet(config.DAILY_REPORT_WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        report_ws = sheet.add_worksheet(title=config.DAILY_REPORT_WORKSHEET_NAME, rows=1000, cols=10)
        report_ws.append_row(["Date", "Category", "Bag Count", "Weight per Bag (kg)", "Total Tonnes"])

    print(f"\nDaily Report for {target_date}")
    print("-" * 50)
    report_rows = []
    grand_total_tonnes = 0.0
    for category, count in sorted(counts.items()):
        weight_kg = config.PRODUCT_WEIGHT_KG.get(category, config.DEFAULT_WEIGHT_KG)
        tonnes = (count * weight_kg) / 1000
        grand_total_tonnes += tonnes
        report_rows.append([target_date, category, count, weight_kg, round(tonnes, 3)])
        print(f"  {category:<15} {count:>5} bags   x {weight_kg}kg  =  {tonnes:.3f} t")

    print("-" * 50)
    print(f"  {'TOTAL':<15} {'':>5}        {'':>10}     {grand_total_tonnes:.3f} t\n")

    report_ws.append_rows(report_rows, value_input_option="USER_ENTERED")
    print(f"Saved {len(report_rows)} category row(s) to '{config.DAILY_REPORT_WORKSHEET_NAME}' worksheet.")


if __name__ == "__main__":
    main()
