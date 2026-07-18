"""
Bag Counter — main pipeline.

# Copyright (c) 2026 Tej Prakash Jangid. All Rights Reserved. Proprietary and Confidential.

Built for: a nozzle fills a bag over ~20-35 seconds, then it's thrown down
onto the floor. Process is slow enough that each bag stays visible in
frame for a solid stretch of time, so we count it once its classification
has been consistent for several consecutive checks -- no need to catch
it mid-motion or rely on a precise crossing line.

Flow per frame:
  1. Read a frame from the camera (RTSP CCTV or phone IP-cam stream).
  2. Background subtraction -> motion blob (the bag region).
  3. Feed the blob's bounding box into the centroid tracker (assigns/keeps
     a persistent track ID for "the bag currently on the floor/in frame").
  4. Periodically classify the tracked blob with CLIP against reference
     category embeddings.
  5. Once a track has held the SAME classification for MIN_STABLE_FRAMES
     consecutive checks AND has existed for at least MIN_TRACK_AGE_TO_COUNT
     frames, and hasn't already been counted -> log one count event.

Run:
    python main.py
Press 'q' in the video window to stop.

Run daily_report.py once every 24 hours (e.g. via a scheduled task at
midnight) to get a same-day tonnage summary per category.
"""
import json
from collections import defaultdict, deque

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import config
from tracker import CentroidTracker
from sheets_logger import SheetsLogger
from mjpeg_stream import MJPEGCapture

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"


def load_reference_embeddings():
    """
    product_embeddings.json stores multiple prototype vectors per category
    (one per K-means cluster from reference_embeddings.py):
        {"GOLD": [[v1], [v2]], "SAK_40KG": [[v1], [v2], [v3]]}
    Flatten into one matrix of all prototypes + a parallel list of which
    category each row belongs to, so matching is "closest prototype wins".
    """
    with open(config.EMBEDDINGS_FILE) as f:
        raw = json.load(f)

    row_names, rows = [], []
    for category, prototype_list in raw.items():
        for vec in prototype_list:
            rows.append(vec)
            row_names.append(category)

    matrix = np.array(rows)  # (total_prototypes, dim), already L2-normalized
    return row_names, matrix


def classify_crop(model, processor, crop_bgr, ref_names, ref_matrix):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    feats = feats.squeeze(0).cpu().numpy()

    sims = ref_matrix @ feats
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score < config.SIMILARITY_THRESHOLD:
        return None, best_score
    return ref_names[best_idx], best_score


def match_boxes_to_tracks(objects, rects):
    """Nearest-centroid match so we know which bounding box belongs to which track ID."""
    box_by_id = {}
    for (x, y, w, h) in rects:
        cx, cy = x + w // 2, y + h // 2
        best_id, best_dist = None, float("inf")
        for oid, centroid in objects.items():
            d = (centroid[0] - cx) ** 2 + (centroid[1] - cy) ** 2
            if d < best_dist:
                best_dist, best_id = d, oid
        if best_id is not None:
            box_by_id[best_id] = (x, y, w, h)
    return box_by_id


def main():
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    print("Loading reference embeddings...")
    ref_names, ref_matrix = load_reference_embeddings()
    print(f"Loaded categories: {sorted(set(ref_names))}")

    print("Connecting to Google Sheets...")
    logger = SheetsLogger()

    print(f"Opening camera stream: {config.RTSP_URL}")
    if config.RTSP_URL.lower().startswith("http"):
        cap = MJPEGCapture(config.RTSP_URL)
    else:
        cap = cv2.VideoCapture(config.RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera stream. Check RTSP_URL / credentials / network.")

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=config.BG_HISTORY, varThreshold=config.BG_VAR_THRESHOLD, detectShadows=False
    )
    tracker = CentroidTracker(max_disappeared=config.MAX_DISAPPEARED_FRAMES)

    track_votes = defaultdict(lambda: deque(maxlen=config.MIN_STABLE_FRAMES))
    counted_tracks = set()
    running_counts = defaultdict(int)
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] frame read failed, retrying...")
                continue
            frame_idx += 1

            fgmask = bg_subtractor.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            fgmask = cv2.dilate(fgmask, np.ones((7, 7), np.uint8), iterations=2)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= config.MIN_CONTOUR_AREA]

            objects, ages = tracker.update(rects)
            box_by_id = match_boxes_to_tracks(objects, rects)

            # Only run the (relatively expensive) CLIP classification every
            # few frames per track -- since a bag sits for 20-35 seconds,
            # we don't need to check it on every single frame.
            should_classify = (frame_idx % config.CLASSIFY_EVERY_N_FRAMES == 0)

            for object_id, (x, y, w, h) in box_by_id.items():
                if object_id in counted_tracks:
                    continue  # already logged this bag, skip re-checking it

                crop = frame[max(0, y):y + h, max(0, x):x + w]
                if crop.size == 0:
                    continue

                label_display = "..."
                if should_classify:
                    category, score = classify_crop(model, processor, crop, ref_names, ref_matrix)
                    track_votes[object_id].append(category)
                    label_display = f"{category or '...'} ({score:.2f})"

                    if (
                        ages.get(object_id, 0) >= config.MIN_TRACK_AGE_TO_COUNT
                        and len(track_votes[object_id]) == config.MIN_STABLE_FRAMES
                    ):
                        votes = [v for v in track_votes[object_id] if v is not None]
                        if votes and votes.count(votes[0]) == len(votes):
                            # unanimous stable vote across MIN_STABLE_FRAMES checks -> count it once
                            category = votes[0]
                            running_counts[category] += 1
                            logger.log_event(category, object_id, running_counts[category])
                            counted_tracks.add(object_id)
                            weight = config.PRODUCT_WEIGHT_KG.get(category, config.DEFAULT_WEIGHT_KG)
                            print(f"[COUNT] {category} bag #{running_counts[category]} "
                                  f"(track {object_id}, {weight}kg)")

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"#{object_id} {label_display}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            y0 = 25
            for category, count in running_counts.items():
                weight = config.PRODUCT_WEIGHT_KG.get(category, config.DEFAULT_WEIGHT_KG)
                tonnes = (count * weight) / 1000
                cv2.putText(frame, f"{category}: {count} bags ({tonnes:.2f} t)", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y0 += 25

            cv2.imshow("Bag Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.stop()


if __name__ == "__main__":
    main()
