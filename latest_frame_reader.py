"""
Decouples raw camera capture rate from however fast inference can keep up.

A background thread continuously reads frames from the camera (up to a
target capture rate) and always keeps only the MOST RECENT frame in
memory -- older frames are simply overwritten, never queued. This
prevents a backlog from building up in the RTSP pipe when YOLO inference
is temporarily slower than the incoming frame rate, which otherwise
causes growing latency over time (inference blocking the same thread
that's supposed to be draining the camera's frames).

Also owns reconnect logic, since it's the thread actually talking to
the camera.
"""
import time
import threading


class LatestFrameReader:
    def __init__(self, open_camera_fn, target_capture_fps=15,
                 max_consecutive_failures=30, reconnect_delay_sec=3,
                 on_status=None):
        self._open_camera_fn = open_camera_fn
        self._min_frame_interval = (1.0 / target_capture_fps) if target_capture_fps else 0
        self._max_consecutive_failures = max_consecutive_failures
        self._reconnect_delay_sec = reconnect_delay_sec
        self._on_status = on_status or (lambda msg: None)

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_ok = False
        self._running = True

        self.cap = self._open_camera_fn()
        self._opened_ok = self.cap.isOpened()

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def isOpened(self):
        return self._opened_ok

    def _capture_loop(self):
        consecutive_failures = 0
        last_read_time = 0.0

        while self._running:
            if self._min_frame_interval:
                elapsed = time.time() - last_read_time
                if elapsed < self._min_frame_interval:
                    time.sleep(self._min_frame_interval - elapsed)
            last_read_time = time.time()

            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None

            if ok:
                with self._lock:
                    self._latest_frame = frame
                    self._latest_ok = True
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= self._max_consecutive_failures:
                    self._on_status("Camera stream dropped -- attempting to reconnect...")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    time.sleep(self._reconnect_delay_sec)
                    try:
                        self.cap = self._open_camera_fn()
                        if self.cap.isOpened():
                            self._on_status("Reconnected. Running.")
                            consecutive_failures = 0
                        else:
                            self._on_status("Reconnect failed, retrying...")
                    except Exception as e:
                        self._on_status(f"Reconnect failed, retrying... ({e})")

    def get_latest(self):
        """Returns (ok, frame) for whatever the most recently captured frame is."""
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return self._latest_ok, self._latest_frame.copy()

    def release(self):
        self._running = False
        try:
            self.cap.release()
        except Exception:
            pass
