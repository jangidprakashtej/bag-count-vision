"""
# Copyright (c) 2026 Tej Prakash Jangid. All Rights Reserved. Proprietary and Confidential.

Central configuration for the Bag Counter pipeline.
Edit these values to match your setup.
"""

# --- Camera ---
RTSP_URL = "http://10.34.151.169:8080/video"  # your CCTV RTSP stream
# For a phone via IP Webcam app, just use the plain /video URL -- main.py
# automatically uses a raw MJPEG reader for http:// URLs, no format hint needed:
# RTSP_URL = "http://10.31.125.60:8080/video"

# --- Products / bag categories ---
# Folder structure expected under REFERENCE_IMAGES_DIR:
#   reference_images/
#       GOLD/          img1.jpg img2.jpg ...
#       ISI/           img1.jpg img2.jpg ...
#       GOLD_ISI/      img1.jpg img2.jpg ...
#       SAK_40KG/      img1.jpg img2.jpg ...
REFERENCE_IMAGES_DIR = "reference_images"
PRODUCT_LIST = [
    "GOLD",
    "ISI",
    "GOLD_ISI",
    "SAK_40KG",
    # ... add all your actual bag categories here, must match folder names above
]

# Weight per bag, in kg, keyed by category name (must match PRODUCT_LIST entries).
# Used to convert bag counts into tonnage in the daily report.
PRODUCT_WEIGHT_KG = {
    "GOLD": 25,
    "ISI": 25,
    "GOLD_ISI": 25,
    "SAK_40KG": 40,
    # add any category not listed here will default to DEFAULT_WEIGHT_KG below
}
DEFAULT_WEIGHT_KG = 25

# --- Recognition ---
EMBEDDINGS_FILE = "product_embeddings.json"
SIMILARITY_THRESHOLD = 0.75    # cosine similarity cutoff to accept a match
CLASSIFY_EVERY_N_FRAMES = 15   # ~every 0.5-1 sec at typical webcam FPS -- plenty given a bag sits for 20-35 sec

# --- Motion / blob detection ---
MIN_CONTOUR_AREA = 8000       # ignore small motion blobs (noise, dust, shadows, minor shake)
BG_HISTORY = 500
BG_VAR_THRESHOLD = 60          # higher = less sensitive to small pixel changes (vibration, lighting flicker)

# --- Tracking / counting ---
# A bag is counted once its track has existed a while AND its classification
# has been the same for several consecutive checks in a row (unanimous vote).
MAX_DISAPPEARED_FRAMES = 30    # frames a track can go missing before dropped (forgiving, since bag sits still)
MIN_STABLE_FRAMES = 4          # classification must agree this many consecutive checks before trusted
MIN_TRACK_AGE_TO_COUNT = 20    # a track must be visible this many frames before it's eligible to count

# --- Google Sheets ---
GOOGLE_SERVICE_ACCOUNT_JSON = "service_account.json"
GOOGLE_SHEET_NAME = "Bag Count Log"
GOOGLE_WORKSHEET_NAME = "Sheet1"
DAILY_REPORT_WORKSHEET_NAME = "Daily Report"
SHEETS_FLUSH_INTERVAL_SEC = 5   # batch writes every N seconds instead of per-event
