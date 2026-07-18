# Bag Counter — Camera → Category Recognition → Tonnage Report

Counts packed bags (e.g. GOLD, ISI, GOLD_ISI, SAK_40KG) as they drop off a
nozzle/chute, tags each with its category and a timestamp in a Google
Sheet, and produces a daily tonnage report per category.

## Design notes specific to this use case

- **Stable-track counting, not line-crossing**: a bag takes 20-35 seconds
  to fill and is then thrown down, so it sits in view for a long stretch.
  Rather than requiring a bag to cross a precise virtual line (which
  assumes a fast, predictable trajectory), a bag is counted once it's
  been tracked for a while (`MIN_TRACK_AGE_TO_COUNT`) and its
  classification has agreed across several consecutive checks
  (`MIN_STABLE_FRAMES`) — much more forgiving and accurate at this pace.
- **Classification isn't run every frame**: `CLASSIFY_EVERY_N_FRAMES`
  spaces out the (relatively expensive) CLIP calls, since a bag isn't
  going anywhere for many seconds — no need to check it 30 times a second.
- **Per-category weight**: `PRODUCT_WEIGHT_KG` in `config.py` lets 25kg
  and 40kg (or any other) bag types convert to tonnes correctly.
- **Daily "reset"**: nothing is actually deleted — every log row has its
  own timestamp, so `daily_report.py` just filters rows to a single date
  and totals them. Schedule it once a day (e.g. via cron at midnight) for
  an automatic daily tonnage report per category.

## How it works

```
RTSP frame
   │
   ▼
Background subtraction  →  motion blob(s) = candidate product region(s)
   │
   ▼
Centroid tracker  →  gives each blob a persistent track ID across frames
   │
   ▼
CLIP embedding of the crop  →  compare to per-product "prototype" embeddings
   │                            (built once from your sample images)
   ▼
Stable classification for N consecutive frames on a track that hasn't
been counted yet  →  ONE count event
   │
   ▼
Buffered append to Google Sheet (Timestamp, Product, Track ID, Running Count)
```

No object-detector training or bounding-box labeling required — CLIP's
pretrained visual embeddings do the recognition, using your existing sample
images as reference. This works best when items pass through frame roughly
one at a time (conveyor, hand-held, gate/turnstile style setup), which fits
"products consecutively come in and out of camera."

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Arrange your reference images
```
reference_images/
    product_a/   img1.jpg  img2.jpg  img3.jpg
    product_b/   img1.jpg  img2.jpg
    ...
```
3–10 images per product, different angles/lighting, works well. Folder
names must match `PRODUCT_LIST` in `config.py`.

### 3. Google Sheets access
1. In Google Cloud Console, create a project → enable the **Google Sheets API**.
2. Create a **Service Account**, generate a JSON key, save it as `service_account.json` next to the scripts.
3. Open the JSON file, copy the `client_email` value.
4. Create a Google Sheet named exactly what you set in `config.GOOGLE_SHEET_NAME`, and **share it** with that `client_email` (Editor access).

### 4. Edit `config.py`
Set `RTSP_URL`, `PRODUCT_LIST`, and tune thresholds (see comments inline).

### 5. Build the reference embeddings (run once, and again if you add images)
```bash
python reference_embeddings.py
```

### 6. Run the counter
```bash
python main.py
```
Press `q` in the video window to stop. A live count overlay is shown; a
window titled "Product Counter" will display bounding boxes, track IDs,
and the matched product + confidence.

## Tuning tips

- **SIMILARITY_THRESHOLD** (config.py): raise it if products are getting
  confused with each other; lower it if a real product isn't being
  recognized at all.
- **MIN_CONTOUR_AREA**: raise if small background movement (shadows,
  hands, insects on lens) is triggering false blobs; lower if small
  products aren't being picked up.
- **MIN_STABLE_FRAMES / MIN_TRACK_AGE_TO_COUNT**: raise these for a
  slower-but-more-certain count if you're seeing double counts or
  wrong matches; lower them for faster response on quick items.
- **Camera framing matters more than the model here** — a fixed camera
  angle, consistent lighting, and one item passing at a time will get
  you the most reliable counts with the least tuning.

## Known limitations / where this would need to grow

- Assumes roughly one product visible at a time. Multiple overlapping
  products in frame simultaneously will confuse the centroid tracker —
  for that, swap in a real detector (YOLOv8 fine-tuned on your products)
  in place of background subtraction.
- Background subtraction assumes a mostly static camera and background
  (typical for fixed CCTV). If the camera moves or lighting changes
  drastically (e.g., day/night), retune `BG_HISTORY` / `BG_VAR_THRESHOLD`
  or switch to a lighting-robust detector.
- CLIP is a general-purpose model — for visually near-identical products
  (e.g., same box, different flavor text) it may struggle without more
  discriminative reference images or a fine-tuned classifier.

## Creative extensions (fun follow-ons once the base pipeline works)

- Text-to-speech announcing each counted product out loud.
- A live dashboard (chart.js/React) reading the same Google Sheet.
- Anomaly alert (e.g., Slack/email) if an unexpected product or a
  mismatch count appears.
- Multi-camera merge: tag each Sheet row with a `camera_id` column.
