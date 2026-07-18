"""
A drop-in replacement for cv2.VideoCapture specifically for HTTP MJPEG
streams (like the Android "IP Webcam" app's /video endpoint).

Why this exists: OpenCV's built-in VideoCapture relies on FFmpeg to open
network streams, and FFmpeg is frequently unable to auto-detect plain
HTTP MJPEG streams even though a browser renders them fine. Rather than
fight URL tricks, this reads the raw multipart HTTP stream directly in
Python and decodes each JPEG frame by hand -- much more reliable for
this specific type of source.

Usage is designed to mimic cv2.VideoCapture just enough for main.py:
    cap = MJPEGCapture(url)
    cap.isOpened()
    ok, frame = cap.read()
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # returns 0, caller should fall back
    cap.release()
"""
import urllib.request
import numpy as np
import cv2


class MJPEGCapture:
    def __init__(self, url, timeout=10):
        self.url = url
        self._opened = False
        self._buffer = b""
        try:
            self._stream = urllib.request.urlopen(url, timeout=timeout)
            self._opened = True
        except Exception as e:
            print(f"[ERROR] could not open MJPEG stream: {e}")
            self._stream = None

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened or self._stream is None:
            return False, None
        try:
            # Keep reading chunks until we find one full JPEG frame
            # (marked by the standard JPEG start 0xFFD8 and end 0xFFD9 bytes)
            while True:
                chunk = self._stream.read(4096)
                if not chunk:
                    self._opened = False
                    return False, None
                self._buffer += chunk

                start = self._buffer.find(b"\xff\xd8")
                end = self._buffer.find(b"\xff\xd9")
                if start != -1 and end != -1 and end > start:
                    jpg_bytes = self._buffer[start:end + 2]
                    self._buffer = self._buffer[end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        return True, frame
                    # corrupt frame, keep looping to try the next one
        except Exception as e:
            print(f"[WARN] MJPEG read error: {e}")
            self._opened = False
            return False, None

    def get(self, prop_id):
        # Frame size isn't known ahead of time with this method; main.py
        # already falls back to a default (640x480) when this returns 0.
        return 0

    def release(self):
        self._opened = False
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
