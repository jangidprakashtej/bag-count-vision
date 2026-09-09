"""
Bag Counter — Desktop App

A simple GUI for factory staff: Start/Stop the camera, see a live video
feed with detections drawn on it, see live running counts per category,
and edit settings (camera URL, thresholds, etc.) without touching code.

Run directly with Python:
    python3.11 app.py

Or package into a standalone double-clickable app with PyInstaller
(see README section "Packaging for distribution").
"""
import tkinter as tk
from tkinter import ttk, messagebox
import queue

import cv2
from PIL import Image, ImageTk

from settings_manager import load_settings, save_settings
from detection_engine import BagCounterEngine
from mjpeg_stream import MJPEGCapture
from rtsp_subprocess_capture import RTSPSubprocessCapture


class BagCounterApp:
    def __init__(self, root):
        self.root = root
        self.settings = load_settings()
        self.root.title(f"Bag Counter — {self.settings.get('factory_name', 'Factory')}")
        self.root.geometry("980x640")

        self.engine = None

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self.live_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        self.calibrate_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.live_tab, text="Live")
        self.notebook.add(self.calibrate_tab, text="Calibrate Zone")
        self.notebook.add(self.settings_tab, text="Settings")

        self._build_live_tab()
        self._build_settings_tab()
        self._build_calibrate_tab()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_engine()

    # ---------------- Live tab ----------------

    def _build_live_tab(self):
        top = ttk.Frame(self.live_tab)
        top.pack(fill="x", pady=8, padx=8)

        self.start_btn = ttk.Button(top, text="Start", command=self._start)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(top, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        self.status_label = ttk.Label(top, text="Stopped", foreground="gray")
        self.status_label.pack(side="left", padx=12)

        body = ttk.Frame(self.live_tab)
        body.pack(fill="both", expand=True, padx=8, pady=8)

        # video panel
        self.video_label = ttk.Label(body, background="black")
        self.video_label.pack(side="left", fill="both", expand=True)

        # counts panel
        counts_frame = ttk.Frame(body, width=260)
        counts_frame.pack(side="right", fill="y", padx=(8, 0))
        ttk.Label(counts_frame, text="Running Counts", font=("", 12, "bold")).pack(anchor="w", pady=(0, 6))
        self.counts_text = tk.Text(counts_frame, width=32, height=25, state="disabled")
        self.counts_text.pack(fill="both", expand=True)

    def _start(self):
        self.settings = load_settings()  # pick up any settings changes before starting
        self.engine = BagCounterEngine(self.settings)
        self.engine.start()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_label.config(text="Starting...", foreground="orange")

    def _stop(self):
        if self.engine:
            self.engine.stop()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Stopped", foreground="gray")

    def _poll_engine(self):
        if self.engine is not None:
            # video frame
            try:
                frame = self.engine.frame_queue.get_nowait()
                self._show_frame(frame)
            except queue.Empty:
                pass

            # status messages
            try:
                while True:
                    msg = self.engine.status_queue.get_nowait()
                    self.status_label.config(text=msg, foreground="green" if "Running" in msg else "orange")
                    if msg.startswith("ERROR:"):
                        self.status_label.config(foreground="red")
                        messagebox.showerror("Bag Counter", msg)
                        self._stop()
                    elif "Google Sheets" in msg:
                        # Sheets connectivity warnings/recoveries are shown in
                        # the status label but shouldn't pop up an alert or
                        # stop the camera/detection, which keeps working fine.
                        self.status_label.config(foreground="orange" if "failed" in msg else "green")
            except queue.Empty:
                pass

            # counts
            try:
                latest_counts = None
                while True:
                    latest_counts = self.engine.count_queue.get_nowait()
            except queue.Empty:
                pass
            if latest_counts:
                self._update_counts_display(latest_counts)

        self.root.after(30, self._poll_engine)

    def _show_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        # scale to fit the label area reasonably
        img.thumbnail((680, 560))
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # keep a reference, avoids garbage collection

    def _update_counts_display(self, counts_dict):
        weights = self.settings.get("product_weight_kg", {})
        default_weight = self.settings.get("default_weight_kg", 25)
        lines = []
        total_tonnes = 0.0
        total_bags = 0
        for category, count in sorted(counts_dict.items()):
            weight = weights.get(category, default_weight)
            tonnes = (count * weight) / 1000
            total_tonnes += tonnes
            total_bags += count
            lines.append(f"{category}\n  {count} bags  ({tonnes:.2f} t)\n")
        lines.append(f"\nTOTAL BAGS: {total_bags}")
        lines.append(f"TOTAL TONNAGE: {total_tonnes:.2f} t")

        self.counts_text.config(state="normal")
        self.counts_text.delete("1.0", tk.END)
        self.counts_text.insert(tk.END, "\n".join(lines))
        self.counts_text.config(state="disabled")

    # ---------------- Settings tab ----------------

    def _build_settings_tab(self):
        frame = ttk.Frame(self.settings_tab)
        frame.pack(fill="both", expand=True, padx=16, pady=16)

        self.setting_vars = {}

        fields = [
            ("factory_name", "Factory Name"),
            ("camera_url", "Camera URL (RTSP or http://.../video)"),
            ("yolo_weights_path", "YOLO Model Weights File (.pt)"),
            ("confidence_threshold", "Detection Confidence Threshold (0-1)"),
            ("min_track_age_to_count", "Min Frames Before Counting"),
            ("min_stable_frames", "Min Stable Classification Checks"),
            ("google_sheet_name", "Google Sheet Name"),
            ("google_worksheet_name", "Log Worksheet Tab Name"),
            ("service_account_json", "Service Account JSON Path"),
            ("duplicate_distance_px", "Duplicate Bag Distance (px)"),
            ("duplicate_memory_seconds", "Duplicate Memory Duration (sec)"),
        ]

        for i, (key, label) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=str(self.settings.get(key, "")))
            entry = ttk.Entry(frame, textvariable=var, width=50)
            entry.grid(row=i, column=1, sticky="w", pady=4, padx=(8, 0))
            self.setting_vars[key] = var

        # Count zone gets its own row of 4 fields (x, y, width, height), since
        # it's a group of related fractions rather than a single value.
        zone_row = len(fields)
        ttk.Label(frame, text="Count Zone (fractions 0-1)").grid(row=zone_row, column=0, sticky="w", pady=(12, 4))
        zone_frame = ttk.Frame(frame)
        zone_frame.grid(row=zone_row, column=1, sticky="w", pady=(12, 4), padx=(8, 0))

        zone = self.settings.get("count_zone") or {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
        self.zone_vars = {}
        for j, part in enumerate(["x", "y", "width", "height"]):
            ttk.Label(zone_frame, text=part).grid(row=0, column=j * 2, padx=(0 if j == 0 else 8, 2))
            var = tk.StringVar(value=str(zone.get(part, 0.0 if part in ("x", "y") else 1.0)))
            entry = ttk.Entry(zone_frame, textvariable=var, width=6)
            entry.grid(row=0, column=j * 2 + 1)
            self.zone_vars[part] = var

        zone_hint = ttk.Label(
            frame,
            text="x/y = top-left corner, width/height = size. All as fractions of the frame (0.0-1.0).",
            foreground="gray",
        )
        zone_hint.grid(row=zone_row + 1, column=0, columnspan=2, sticky="w")

        save_btn = ttk.Button(frame, text="Save Settings", command=self._save_settings)
        save_btn.grid(row=zone_row + 2, column=0, columnspan=2, pady=16)

        note = ttk.Label(
            frame,
            text="Note: per-category bag weights (kg) are still edited directly in settings.json for now.",
            foreground="gray",
        )
        note.grid(row=zone_row + 3, column=0, columnspan=2, sticky="w")

    def _save_settings(self):
        for key, var in self.setting_vars.items():
            value = var.get()
            # cast numeric fields back to numbers
            if key in ("confidence_threshold",):
                value = float(value)
            elif key in ("min_track_age_to_count", "min_stable_frames",
                         "duplicate_distance_px", "duplicate_memory_seconds"):
                value = int(value)
            self.settings[key] = value

        try:
            self.settings["count_zone"] = {
                part: float(var.get()) for part, var in self.zone_vars.items()
            }
        except ValueError:
            messagebox.showerror("Bag Counter", "Count zone values must be numbers between 0 and 1.")
            return

        save_settings(self.settings)
        messagebox.showinfo("Bag Counter", "Settings saved. Restart counting for changes to take effect.")

    # ---------------- Calibrate Zone tab ----------------

    CANVAS_W, CANVAS_H = 640, 360  # matches the engine's internal capture resolution 1:1

    def _build_calibrate_tab(self):
        frame = ttk.Frame(self.calibrate_tab)
        frame.pack(fill="both", expand=True, padx=8, pady=8)

        top = ttk.Frame(frame)
        top.pack(fill="x")
        ttk.Button(top, text="Refresh Preview", command=self._refresh_calibrate_preview).pack(side="left", padx=4)
        ttk.Button(top, text="Save Zone", command=self._save_calibrated_zone).pack(side="left", padx=4)
        ttk.Button(top, text="Clear Zone (use full frame)", command=self._clear_calibrated_zone).pack(side="left", padx=4)

        warning = ttk.Label(
            frame,
            text="Note: click 'Stop' on the Live tab first if it's running -- some cameras only "
                 "allow one connection at a time, so calibrating while live monitoring is active "
                 "may fail to grab a preview.",
            foreground="gray", wraplength=700, justify="left",
        )
        warning.pack(anchor="w", pady=(6, 6))

        hint = ttk.Label(frame, text="Click and drag on the image below to draw the count zone.")
        hint.pack(anchor="w", pady=(0, 6))

        self.calibrate_canvas = tk.Canvas(frame, width=self.CANVAS_W, height=self.CANVAS_H, background="black")
        self.calibrate_canvas.pack()
        self.calibrate_canvas.bind("<ButtonPress-1>", self._on_zone_drag_start)
        self.calibrate_canvas.bind("<B1-Motion>", self._on_zone_drag_move)
        self.calibrate_canvas.bind("<ButtonRelease-1>", self._on_zone_drag_end)

        self._zone_rect_id = None
        self._zone_drag_start = None
        self._pending_zone = None
        self._calibrate_photo = None

    def _grab_preview_frame(self):
        settings = load_settings()
        url = settings.get("camera_url", "")
        if not url:
            messagebox.showerror("Bag Counter", "Set a Camera URL in Settings first.")
            return None
        cap = None
        try:
            cap = MJPEGCapture(url) if url.lower().startswith("http") else RTSPSubprocessCapture(url)
            ok, frame = cap.read()
        except Exception as e:
            messagebox.showerror("Bag Counter", f"Could not open camera for preview: {e}")
            return None
        finally:
            if cap is not None:
                cap.release()
        if not ok:
            messagebox.showerror("Bag Counter", "Could not read a frame from the camera.")
            return None
        return frame

    def _refresh_calibrate_preview(self):
        frame = self._grab_preview_frame()
        if frame is None:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((self.CANVAS_W, self.CANVAS_H))
        self._calibrate_photo = ImageTk.PhotoImage(image=img)
        self.calibrate_canvas.delete("all")
        self.calibrate_canvas.create_image(0, 0, anchor="nw", image=self._calibrate_photo)

        # draw the currently-saved zone (if any) so you can see/adjust it
        zone = self.settings.get("count_zone")
        if zone:
            x0 = zone["x"] * self.CANVAS_W
            y0 = zone["y"] * self.CANVAS_H
            x1 = x0 + zone["width"] * self.CANVAS_W
            y1 = y0 + zone["height"] * self.CANVAS_H
            self._zone_rect_id = self.calibrate_canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2)

    def _on_zone_drag_start(self, event):
        self._zone_drag_start = (event.x, event.y)
        if self._zone_rect_id:
            self.calibrate_canvas.delete(self._zone_rect_id)
            self._zone_rect_id = None

    def _on_zone_drag_move(self, event):
        if not self._zone_drag_start:
            return
        if self._zone_rect_id:
            self.calibrate_canvas.delete(self._zone_rect_id)
        x0, y0 = self._zone_drag_start
        self._zone_rect_id = self.calibrate_canvas.create_rectangle(
            x0, y0, event.x, event.y, outline="red", width=2
        )

    def _on_zone_drag_end(self, event):
        if not self._zone_drag_start:
            return
        x0, y0 = self._zone_drag_start
        x1, y1 = event.x, event.y
        left, right = sorted([max(0, min(self.CANVAS_W, v)) for v in (x0, x1)])
        top, bottom = sorted([max(0, min(self.CANVAS_H, v)) for v in (y0, y1)])
        self._zone_drag_start = None

        if right - left < 5 or bottom - top < 5:
            return  # too small, probably an accidental click, ignore

        self._pending_zone = {
            "x": left / self.CANVAS_W,
            "y": top / self.CANVAS_H,
            "width": (right - left) / self.CANVAS_W,
            "height": (bottom - top) / self.CANVAS_H,
        }

    def _save_calibrated_zone(self):
        if not self._pending_zone:
            messagebox.showwarning("Bag Counter", "Draw a zone on the preview first (click and drag).")
            return
        self.settings = load_settings()
        self.settings["count_zone"] = self._pending_zone
        save_settings(self.settings)

        # keep the numeric fields on the Settings tab in sync
        for part, var in self.zone_vars.items():
            var.set(str(round(self._pending_zone[part], 3)))

        z = self._pending_zone
        messagebox.showinfo(
            "Bag Counter",
            f"Zone saved: x={z['x']:.2f}, y={z['y']:.2f}, "
            f"width={z['width']:.2f}, height={z['height']:.2f}"
        )

    def _clear_calibrated_zone(self):
        self.settings = load_settings()
        self.settings["count_zone"] = None
        save_settings(self.settings)
        for part, var in self.zone_vars.items():
            var.set("0.0" if part in ("x", "y") else "1.0")
        if self._zone_rect_id:
            self.calibrate_canvas.delete(self._zone_rect_id)
            self._zone_rect_id = None
        messagebox.showinfo("Bag Counter", "Zone cleared -- the full frame will be used.")

    def _on_close(self):
        if self.engine and self.engine.is_running():
            self.engine.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = BagCounterApp(root)
    root.mainloop()
