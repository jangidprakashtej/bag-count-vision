"""
RTSP capture via a direct ffmpeg subprocess, bypassing OpenCV's own
FFmpeg plugin entirely.

Why this exists: on this Mac, no available opencv-python build both
(a) imports successfully AND (b) has a working FFmpeg plugin for RTSP --
confirmed via direct testing. A standalone static ffmpeg binary (from
evermeet.cx) DOES work correctly though, so this class drives that
binary directly as a subprocess and reads raw decoded frames from its
output pipe, mimicking cv2.VideoCapture's interface (isOpened(), read(),
release()) so the rest of the app doesn't need to know the difference.
"""
import os
import shutil
import subprocess
import numpy as np

FRAME_WIDTH = 640
FRAME_HEIGHT = 360  # 16:9, matches the camera's actual aspect ratio -- avoids squishing/distorting bag shapes
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3  # bgr24 = 3 bytes/pixel


import platform


def _find_ffmpeg_binary():
    exe_name = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
    # Prefer a copy sitting next to this script (e.g. in the project folder)
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), exe_name)
    if os.path.exists(local):
        return local
    # Fall back to whatever's on PATH (e.g. if moved to ~/bin, or Windows PATH)
    found = shutil.which(exe_name)
    if found:
        return found
    raise RuntimeError(
        f"No {exe_name} binary found. Place a working '{exe_name}' executable "
        f"next to the app's Python files, or install it and make sure it's on PATH."
    )


class RTSPSubprocessCapture:
    def __init__(self, rtsp_url, ffmpeg_path=None):
        self.ffmpeg_path = ffmpeg_path or _find_ffmpeg_binary()
        self._opened = False
        self._proc = None
        self._open(rtsp_url)

    def _open(self, rtsp_url):
        cmd = [
            self.ffmpeg_path,
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-loglevel", "error",
            "-an",
            "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=FRAME_SIZE * 2,
        )
        self._opened = True

    def isOpened(self):
        return self._opened and self._proc is not None and self._proc.poll() is None

    def read(self):
        if not self.isOpened():
            return False, None
        raw = self._proc.stdout.read(FRAME_SIZE)
        if len(raw) != FRAME_SIZE:
            self._opened = False
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3)).copy()
        return True, frame

    def get(self, prop_id):
        return 0

    def release(self):
        self._opened = False
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
