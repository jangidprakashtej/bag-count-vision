"""
Settings are stored in settings.json (created automatically with defaults
on first run) instead of config.py, so the GUI's Settings screen can
read and write them directly -- each factory installation gets its own
settings.json sitting next to the app with that site's camera URL,
Sheet name, etc.
"""
import json
import os

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

DEFAULT_SETTINGS = {
    "camera_url": "http://192.168.1.42:8080/video",
    "yolo_weights_path": "best.pt",
    "confidence_threshold": 0.5,
    "min_track_age_to_count": 20,
    "min_stable_frames": 4,
    "max_disappeared_frames": 30,
    "product_weight_kg": {
        "Gold ISI": 25,
        "Gold NISI": 25,
        "Gold Plus": 25,
        "Sak 25kg ISI": 25,
        "Sak 25kg NISI": 25,
        "Suntara Gypsum 25kg": 25,
        "Suntara Gypsum 30kg": 30,
    },
    "default_weight_kg": 25,
    "google_sheet_name": "Bag Count Log",
    "google_worksheet_name": "Sheet1",
    "daily_report_worksheet_name": "Daily Report",
    "qc_worksheet_name": "QC Samples",
    "service_account_json": "service_account.json",
    "sheets_flush_interval_sec": 5,
    "factory_name": "Factory 1",
    "count_zone": None,
    "count_line_a": None,   # e.g. {"axis": "horizontal", "position": 0.35} -- REQUIRED for counting to work
    "count_line_b": None,   # e.g. {"axis": "horizontal", "position": 0.55} -- REQUIRED for counting to work
    "max_seconds_between_lines": 15,
    "capture_target_fps": 15,
    "inference_target_fps": 6,
    "duplicate_distance_px": 60,
    "duplicate_memory_seconds": 12,  # bridges brief occlusion only -- must be well UNDER your actual seconds-per-bag cycle time, or legitimately new bags landing in the same spot get wrongly skipped
}


def load_settings():
    if not os.path.exists(SETTINGS_PATH):
        save_settings(DEFAULT_SETTINGS)
        return dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, "r") as f:
            loaded = json.load(f)
        # fill in any missing keys with defaults (handles upgrades gracefully)
        merged = dict(DEFAULT_SETTINGS)
        merged.update(loaded)
        return merged
    except Exception as e:
        print(f"[WARN] could not load settings.json ({e}), using defaults")
        return dict(DEFAULT_SETTINGS)


def save_settings(settings_dict):
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings_dict, f, indent=2)
