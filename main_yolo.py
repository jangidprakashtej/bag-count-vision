"""
Bag Counter — YOLOv8 version.

# Copyright (c) 2026 Tej Prakash Jangid. All Rights Reserved. Proprietary and Confidential.

Replaces background-subtraction + CLIP with a custom-trained YOLOv8
detector (see train_yolo.py). This directly finds AND classifies each
bag in frame, handling busy backgrounds / multiple bags naturally, and
is not fooled by camera shake or lighting flicker the way motion-based
background subtraction was -- it only reacts to things that actually
look like a trained bag category.

Counting logic (stable-track confirmation) stays the same as before,
just fed by YOLO detections instead of motion blobs + CLIP matching.

Run:
    python3.11 main_yolo.py
Press 'q' in the video window to stop.
"""
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

import config
from tracker import CentroidTracker
from sheets_logger import SheetsLogger
from mjpeg_stream import MJPEGCapture

YOLO_WEIGHTS_PATH = "bag_detector_training/run1/weights/best.pt"  # from train_yolo.py
YOLO_CONFIDENCE_THRESHOLD = 0.5


def match_boxes_to_tracks(objects, rects):
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
    print("Loading YOLOv8 bag detector...")
    model = YOLO(YOLO_WEIGHTS_PATH)
    category_names = model.names  # {0: "Gold ISI", 1: "Sak 25kg ISI", ...} from your data.yaml
    print(f"Loaded categories: {list(category_names.values())}")

    print("Connecting to Google Sheets...")
    logger = SheetsLogger()

    print("Resuming today's counts from the Sheet (if any)...")
    resumed_counts = logger.get_todays_counts()
    if resumed_counts:
        print(f"Resumed: {resumed_counts}")

    print(f"Opening camera stream: {config.RTSP_URL}")
    if config.RTSP_URL.lower().startswith("http"):
        cap = MJPEGCapture(config.RTSP_URL)
    else:
        cap = cv2.VideoCapture(config.RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera stream. Check RTSP_URL / credentials / network.")

    tracker = CentroidTracker(max_disappeared=config.MAX_DISAPPEARED_FRAMES)
    track_votes = defaultdict(lambda: deque(maxlen=config.MIN_STABLE_FRAMES))
    counted_tracks = set()
    running_counts = defaultdict(int, resumed_counts)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] frame read failed, retrying...")
                continue

            results = model.predict(frame, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)[0]

            rects = []
            box_categories = {}  # (x, y, w, h) -> category name, matched up after tracking
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                category = category_names[int(box.cls[0])]
                rects.append((x, y, w, h))
                box_categories[(x, y, w, h)] = category

            objects, ages = tracker.update(rects)
            box_by_id = match_boxes_to_tracks(objects, rects)

            for object_id, (x, y, w, h) in box_by_id.items():
                category = box_categories.get((x, y, w, h))
                if object_id in counted_tracks or category is None:
                    continue

                track_votes[object_id].append(category)

                if (
                    ages.get(object_id, 0) >= config.MIN_TRACK_AGE_TO_COUNT
                    and len(track_votes[object_id]) == config.MIN_STABLE_FRAMES
                ):
                    votes = list(track_votes[object_id])
                    if votes.count(votes[0]) == len(votes):
                        final_category = votes[0]
                        running_counts[final_category] += 1
                        logger.log_event(final_category, object_id, running_counts[final_category])
                        counted_tracks.add(object_id)
                        weight = config.PRODUCT_WEIGHT_KG.get(final_category, config.DEFAULT_WEIGHT_KG)
                        print(f"[COUNT] {final_category} bag #{running_counts[final_category]} "
                              f"(track {object_id}, {weight}kg)")

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"#{object_id} {category}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            y0 = 25
            for cat, count in running_counts.items():
                weight = config.PRODUCT_WEIGHT_KG.get(cat, config.DEFAULT_WEIGHT_KG)
                tonnes = (count * weight) / 1000
                cv2.putText(frame, f"{cat}: {count} bags ({tonnes:.2f} t)", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y0 += 25

            cv2.imshow("Bag Counter (YOLOv8)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.stop()


if __name__ == "__main__":
    main()
