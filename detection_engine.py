"""
Core bag-counting logic.

Architecture:
  - A background LatestFrameReader thread captures frames from the camera
    at up to ~capture_target_fps (default 15), always keeping only the
    MOST RECENT frame -- decoupled from inference speed, so a slow
    inference cycle never causes an RTSP backlog to build up.
  - The main loop here runs INFERENCE at its own, slower pace
    (inference_target_fps, default ~6), pulling whatever the latest
    available frame is each cycle.
  - Tracking: Ultralytics' built-in ByteTrack (motion-prediction based,
    much more robust to brief occlusion than simple centroid matching).
  - Counting: DUAL VIRTUAL LINE CROSSING. A bag is only counted once it
    crosses count_line_a, THEN count_line_b, within max_seconds_between_lines
    of each other. This directly solves the "sitting for hours" and
    "nudged slightly by a person" false-recount problems -- a stationary
    or barely-moved bag never crosses a line a second time, so it can
    never be recounted, no matter how long it sits there.
  - A position-based duplicate memory remains as an extra safety net on
    top of line-crossing + ByteTrack, catching rare edge cases.
  - Bounding boxes are held on-screen for a short grace period even if a
    specific inference cycle momentarily misses the object, to avoid
    visual flicker.

IMPORTANT: count_line_a and count_line_b must be configured in settings
for ANY counting to happen -- with no lines set, bags are still detected
and tracked (visible on screen) but nothing gets counted, by design.
"""
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import time
import threading
import queue
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

from sheets_logger import SheetsLogger
from mjpeg_stream import MJPEGCapture
from rtsp_subprocess_capture import RTSPSubprocessCapture
from latest_frame_reader import LatestFrameReader

INFERENCE_RESIZE_WIDTH = 640
BOX_HOLD_SECONDS = 1.5   # keep drawing a box this long after its last real detection, to avoid flicker


def _in_count_zone(centroid, zone, frame_w, frame_h):
    if not zone:
        return True
    zx = zone.get("x", 0.0) * frame_w
    zy = zone.get("y", 0.0) * frame_h
    zw = zone.get("width", 1.0) * frame_w
    zh = zone.get("height", 1.0) * frame_h
    cx, cy = centroid
    return zx <= cx <= zx + zw and zy <= cy <= zy + zh


def _distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _crossed_line(prev_centroid, curr_centroid, line, frame_w, frame_h):
    """line: {"axis": "horizontal"|"vertical", "position": 0.0-1.0} or None."""
    if not line or prev_centroid is None:
        return False
    axis = line.get("axis", "horizontal")
    if axis == "horizontal":
        pos_px = line.get("position", 0.5) * frame_h
        prev_val, curr_val = prev_centroid[1], curr_centroid[1]
    else:
        pos_px = line.get("position", 0.5) * frame_w
        prev_val, curr_val = prev_centroid[0], curr_centroid[0]
    return (prev_val < pos_px <= curr_val) or (prev_val > pos_px >= curr_val)


class BagCounterEngine:
    def __init__(self, settings):
        self.settings = settings
        self.frame_queue = queue.Queue(maxsize=2)
        self.count_queue = queue.Queue()
        self.status_queue = queue.Queue()
        self.latest_counts = {}   # always-current snapshot, readable directly (avoids any queue-draining race)
        self._thread = None
        self._stop_flag = threading.Event()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def _emit_status(self, message):
        self.status_queue.put(message)
        print(message)

    def _open_camera(self, s):
        url = s["camera_url"]
        if url.lower().startswith("http"):
            return MJPEGCapture(url)
        if url.lower().startswith("rtsp"):
            return RTSPSubprocessCapture(url)
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    def _resize_for_inference(self, frame):
        h, w = frame.shape[:2]
        if w <= INFERENCE_RESIZE_WIDTH:
            return frame, 1.0
        scale = INFERENCE_RESIZE_WIDTH / w
        small = cv2.resize(frame, (INFERENCE_RESIZE_WIDTH, int(h * scale)))
        return small, scale

    def _run(self):
        s = self.settings
        try:
            self._emit_status("Loading YOLOv8 model...")
            model = YOLO(s["yolo_weights_path"])
            category_names = model.names

            self._emit_status("Connecting to Google Sheets...")
            logger = SheetsLogger(s, on_error=self._emit_status)

            self._emit_status("Resuming today's counts...")
            resumed_counts = logger.get_todays_counts()

            capture_fps = s.get("capture_target_fps", 15)
            inference_fps = s.get("inference_target_fps", 6)
            inference_interval = 1.0 / inference_fps if inference_fps else 0

            self._emit_status(f"Opening camera: {s['camera_url']}")
            reader = LatestFrameReader(
                lambda: self._open_camera(s),
                target_capture_fps=capture_fps,
                on_status=self._emit_status,
            )
            if not reader.isOpened():
                self._emit_status("ERROR: could not open camera stream. Check URL/network in Settings.")
                return

            line_a = s.get("count_line_a")   # {"axis": "horizontal"/"vertical", "position": 0.0-1.0}
            line_b = s.get("count_line_b")
            if not line_a or not line_b:
                self._emit_status(
                    "WARNING: count_line_a / count_line_b not configured -- "
                    "bags will be detected and tracked but NOTHING will be counted "
                    "until both lines are set in Settings."
                )

            max_seconds_between_lines = s.get("max_seconds_between_lines", 15)
            count_zone = s.get("count_zone")

            running_counts = defaultdict(int, resumed_counts)
            if resumed_counts:
                self.latest_counts = dict(running_counts)
                self.count_queue.put(dict(running_counts))  # show resumed totals immediately, don't wait for the next new count
            track_votes = defaultdict(lambda: deque(maxlen=s.get("min_stable_frames", 4)))
            track_prev_centroid = {}
            track_crossed_a_time = {}
            counted_tracks = set()

            # position-based duplicate memory -- extra safety net
            counted_positions = []
            duplicate_skipped_tracks = set()
            duplicate_distance_px = s.get("duplicate_distance_px", 60)
            duplicate_memory_seconds = s.get("duplicate_memory_seconds", 12)

            # box-hold state, to smooth over momentary missed detections
            box_last_seen = {}   # track_id -> {"rect": (x,y,w,h), "category": str, "time": t}

            self._emit_status("Running.")
            last_inference_time = 0.0

            while not self._stop_flag.is_set():
                ok, frame = reader.get_latest()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                now = time.time()

                if inference_interval and (now - last_inference_time) < inference_interval:
                    # Not time for another inference cycle -- still show the
                    # live frame (with held boxes + lines) so video doesn't
                    # look frozen, just skip re-running the model.
                    self._draw_held_boxes(frame, box_last_seen, counted_tracks, duplicate_skipped_tracks)
                    self._draw_lines(frame, line_a, line_b)
                    self._push_frame(frame)
                    time.sleep(0.01)
                    continue
                last_inference_time = now

                small_frame, scale = self._resize_for_inference(frame)
                results = model.track(
                    small_frame,
                    conf=s["confidence_threshold"],
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )[0]

                frame_h, frame_w = frame.shape[:2]

                if results.boxes.id is not None:
                    for box in results.boxes:
                        track_id = int(box.id[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        category = category_names[int(box.cls[0])]
                        centroid = (x + w / 2, y + h / 2)

                        box_last_seen[track_id] = {"rect": (x, y, w, h), "category": category, "time": now}
                        prev_centroid = track_prev_centroid.get(track_id)

                        if track_id not in counted_tracks:
                            track_votes[track_id].append(category)

                            crossed_a_now = _crossed_line(prev_centroid, centroid, line_a, frame_w, frame_h)
                            crossed_b_now = _crossed_line(prev_centroid, centroid, line_b, frame_w, frame_h)

                            if crossed_a_now and track_id not in track_crossed_a_time:
                                track_crossed_a_time[track_id] = now
                                print(f"[LINE A] track {track_id} crossed line A ({category})")

                            ready_to_count = False
                            if crossed_b_now and track_id in track_crossed_a_time:
                                if now - track_crossed_a_time[track_id] <= max_seconds_between_lines:
                                    ready_to_count = True
                                    print(f"[LINE B] track {track_id} crossed line B -- eligible to count")
                                else:
                                    print(f"[LINE B] track {track_id} crossed line B but too long after line A "
                                          f"({now - track_crossed_a_time[track_id]:.1f}s > {max_seconds_between_lines}s limit) -- resetting")
                                    del track_crossed_a_time[track_id]

                            if ready_to_count:
                                votes = list(track_votes[track_id])
                                final_category = max(set(votes), key=votes.count) if votes else category

                                counted_positions[:] = [
                                    p for p in counted_positions
                                    if now - p["time"] < duplicate_memory_seconds
                                ]
                                is_duplicate = any(
                                    _distance(centroid, p["centroid"]) < duplicate_distance_px
                                    for p in counted_positions
                                )

                                counted_tracks.add(track_id)

                                if not is_duplicate:
                                    running_counts[final_category] += 1
                                    logger.log_event(final_category, track_id, running_counts[final_category])
                                    counted_positions.append({"centroid": centroid, "time": now})
                                    self.latest_counts = dict(running_counts)
                                    self.count_queue.put(dict(running_counts))
                                    print(f"[COUNT] {final_category} bag #{running_counts[final_category]} (track {track_id})")
                                else:
                                    duplicate_skipped_tracks.add(track_id)
                                    print(f"[SKIP] track {track_id} looks like a bag already counted nearby -- not recounting")

                        track_prev_centroid[track_id] = centroid

                # forget box-hold entries that are stale well beyond the hold window
                stale_ids = [tid for tid, info in box_last_seen.items() if now - info["time"] > BOX_HOLD_SECONDS * 4]
                for tid in stale_ids:
                    del box_last_seen[tid]

                self._draw_held_boxes(frame, box_last_seen, counted_tracks, duplicate_skipped_tracks)
                self._draw_lines(frame, line_a, line_b)
                self._push_frame(frame)

            reader.release()
            logger.stop()
            self._emit_status("Stopped.")

        except Exception as e:
            self._emit_status(f"ERROR: {e}")

    def _draw_held_boxes(self, frame, box_last_seen, counted_tracks, duplicate_skipped_tracks):
        now = time.time()
        for track_id, info in box_last_seen.items():
            if now - info["time"] > BOX_HOLD_SECONDS:
                continue
            x, y, w, h = info["rect"]
            if track_id in duplicate_skipped_tracks:
                label, color = "SKIPPED (duplicate)", (0, 165, 255)
            elif track_id in counted_tracks:
                label, color = "counted", (0, 255, 0)
            else:
                label, color = info["category"], (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"#{track_id} {label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_lines(self, frame, line_a, line_b):
        h, w = frame.shape[:2]
        for line, color, name in [(line_a, (255, 0, 0), "A"), (line_b, (0, 0, 255), "B")]:
            if not line:
                continue
            axis = line.get("axis", "horizontal")
            pos = line.get("position", 0.5)
            if axis == "horizontal":
                y = int(pos * h)
                cv2.line(frame, (0, y), (w, y), color, 2)
                cv2.putText(frame, name, (5, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                x = int(pos * w)
                cv2.line(frame, (x, 0), (x, h), color, 2)
                cv2.putText(frame, name, (x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _push_frame(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
        else:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put(frame)
