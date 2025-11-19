"""
camera_control.py - Camera Parameter Automation & Logging Tool

A production-ready tool for controlling camera parameters (brightness, contrast, exposure)
with real-time preview, CSV logging, and snapshot capture capabilities.

Dependencies:
    pip install opencv-python numpy

Usage Examples:
    # Run with default settings (camera 0)
    python camera_control.py

    # Specify camera and custom log/snapshot locations
    python camera_control.py --camera 0 --log logs/params_log.csv --save-dir snapshots

    # Run in simulation mode (no physical camera needed)
    python camera_control.py --simulate --verbose

    # Preview recent log entries
    python camera_control.py --preview-log

Keyboard Controls:
    q - Quit application
    s - Save snapshot with parameter overlay
    r - Reset parameters to defaults
    Arrow Up/Down - Adjust brightness
    Arrow Left/Right - Adjust contrast

License: MIT
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import traceback
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DEFAULT_BRIGHTNESS = 128
DEFAULT_CONTRAST = 128
DEFAULT_EXPOSURE = 128  # slider center
WINDOW_NAME = "Camera Control"

USAGE_EXAMPLES = """
USAGE EXAMPLES:

1. Basic usage with default camera:
   python camera_control.py

2. Custom camera and paths:
   python camera_control.py --camera 0 --log logs/params_log.csv --save-dir snapshots

3. Simulation mode (for testing without camera):
   python camera_control.py --simulate --verbose

4. Preview recent log entries:
   python camera_control.py --preview-log
"""

README_SNIPPET = """
# Camera Parameter Automation & Logging Tool

This tool provides real-time control and logging of camera parameters including brightness,
contrast, and exposure. It features an intuitive trackbar interface, automatic CSV logging
of all parameter changes, and the ability to capture annotated snapshots.

## Key Features
- Real-time camera feed with adjustable parameters via trackbars
- CSV logging of all parameter changes with timestamps
- Snapshot capture with parameter overlay and automatic naming
- Simulation mode for testing without physical camera
- Cross-platform support (Windows, Linux, macOS)
"""

CSV_HEADER = ["timestamp", "camera_index", "brightness", "contrast", "exposure", "action", "note"]


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create directory (and parents) if not exists."""
    if not path:
        return
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        # If creation fails, let callers handle errors later
        pass


def iso_timestamp() -> str:
    """Return ISO-like timestamp with milliseconds: YYYY-MM-DD HH:MM:SS.mmm"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def iso_filename_timestamp() -> str:
    """Return compact timestamp suitable for filenames: YYYYMMDD_HHMMSS"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# CSV LOGGING
# ---------------------------------------------------------------------------

def init_log(csv_path: str) -> None:
    """Initialize CSV log file with header if it doesn't exist."""
    dirpath = os.path.dirname(csv_path)
    ensure_dir(dirpath)
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


def log_params(csv_path: str, camera_index: int, brightness: int, contrast: int,
               exposure: float, action: str, note: str = "") -> None:
    """Append a single row to the CSV log."""
    ensure_dir(os.path.dirname(csv_path))
    ts = iso_timestamp()
    row = [ts, camera_index, brightness, contrast, f"{exposure:.3f}", action, note]
    try:
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        # If logging fails, print but continue
        print(f"[WARN] Failed to write log row: {e}")


# ---------------------------------------------------------------------------
# PARAMETER HELPERS
# ---------------------------------------------------------------------------

def apply_brightness_contrast(frame: np.ndarray, brightness: int, contrast: int) -> np.ndarray:
    """
    Apply brightness and contrast to an image.

    brightness: 0..255 -> mapped to beta = brightness - 128
    contrast: 0..255 -> mapped to alpha = contrast / 128.0 (128 => 1.0)
    """
    beta = int(brightness) - 128
    alpha = float(contrast) / 128.0
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return adjusted


def map_exposure_slider_to_value(slider_val: int) -> float:
    """
    Map slider (0..255) to an exposure value.

    Many industrial cameras use negative values for exposure on some drivers (e.g. -13 .. -1)
    or milliseconds for others. We map slider -> [-13.0, -1.0] as a sensible default
    for demonstration; this can be adjusted for a specific camera SDK.
    """
    s = max(0, min(255, int(slider_val)))
    exposure = (s / 255.0) * ( -1.0 - (-13.0) ) + (-13.0)  # maps 0->-13, 255->-1
    return float(exposure)


# ---------------------------------------------------------------------------
# UI: trackbars and reading values
# ---------------------------------------------------------------------------

def nothing(_: int) -> None:
    """No-op for trackbar callback."""
    pass


def setup_ui(window_name: str) -> None:
    """Create named window and create trackbars with default neutral positions."""
    # Use WINDOW_NORMAL so user can resize
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Brightness", window_name, DEFAULT_BRIGHTNESS, 255, nothing)
    cv2.createTrackbar("Contrast", window_name, DEFAULT_CONTRAST, 255, nothing)
    cv2.createTrackbar("Exposure", window_name, DEFAULT_EXPOSURE, 255, nothing)


def read_trackbars(window_name: str) -> Tuple[int, int, int]:
    """Return (brightness, contrast, exposure_slider)."""
    b = cv2.getTrackbarPos("Brightness", window_name)
    c = cv2.getTrackbarPos("Contrast", window_name)
    e = cv2.getTrackbarPos("Exposure", window_name)
    return int(b), int(c), int(e)


def reset_trackbars(window_name: str) -> None:
    """Reset trackbars to default neutral values."""
    cv2.setTrackbarPos("Brightness", window_name, DEFAULT_BRIGHTNESS)
    cv2.setTrackbarPos("Contrast", window_name, DEFAULT_CONTRAST)
    cv2.setTrackbarPos("Exposure", window_name, DEFAULT_EXPOSURE)


# ---------------------------------------------------------------------------
# SNAPSHOT / SIMULATION / LOG PREVIEW
# ---------------------------------------------------------------------------

def save_snapshot(frame: np.ndarray, snapshot_dir: str, brightness: int, contrast: int,
                  exposure: float) -> str:
    """
    Annotate and save snapshot. Returns full filepath.
    """
    ensure_dir(snapshot_dir)
    fname = f"snap_{iso_filename_timestamp()}_b{brightness}_c{contrast}_e{exposure:.1f}.png"
    filepath = os.path.join(snapshot_dir, fname)

    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_y = 28
    line_h = 28
    cv2.putText(annotated, f"Brightness: {brightness}", (10, base_y), font, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"Contrast:   {contrast}", (10, base_y + line_h), font, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"Exposure:   {exposure:.1f}", (10, base_y + 2 * line_h), font, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, iso_timestamp(), (10, base_y + 3 * line_h), font, 0.5, (0, 255, 0), 1)

    # Attempt to write image; catch errors
    try:
        cv2.imwrite(filepath, annotated)
    except Exception as e:
        print(f"[WARN] Failed to save snapshot: {e}")
        return ""

    return filepath


def preview_log(csv_path: str, num_lines: int = 5) -> None:
    """Print last num_lines from log (excluding header)."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] Log file not found: {csv_path}")
        return
    try:
        with open(csv_path, "r", newline="") as f:
            reader = list(csv.reader(f))
    except Exception as e:
        print(f"[ERROR] Could not read log: {e}")
        return

    if len(reader) <= 1:
        print("[INFO] Log file contains only header or is empty.")
        return

    header = reader[0]
    rows = reader[1:]
    tail = rows[-num_lines:]

    print("\n" + "=" * 80)
    print(f"LAST {len(tail)} LOG ENTRIES FROM: {csv_path}")
    print("=" * 80)
    print(" | ".join(header))
    print("-" * 80)
    for r in tail:
        print(" | ".join(r))
    print("=" * 80 + "\n")


def create_test_frame(width: int, height: int, frame_number: int) -> np.ndarray:
    """Create a colorful synthetic frame for simulation mode."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # vertical gradient
    for i in range(height):
        val = int(255 * (i / height))
        frame[i, :] = (val, 100, 255 - val)
    # overlay text and shapes
    txt = f"SIMULATION MODE - Frame {frame_number}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    tsize = cv2.getTextSize(txt, font, 1, 2)[0]
    x = max(10, (width - tsize[0]) // 2)
    y = max(30, (height + tsize[1]) // 2)
    cv2.putText(frame, txt, (x, y), font, 1, (255, 255, 255), 2)
    cv2.circle(frame, (width // 4, height // 4), 40, (255, 0, 0), -1)
    cv2.rectangle(frame, (3 * width // 4 - 40, height // 4 - 40), (3 * width // 4 + 40, height // 4 + 40), (0, 255, 0), -1)
    return frame


# ---------------------------------------------------------------------------
# PARAM TRACKER CLASS
# ---------------------------------------------------------------------------

class ParamTracker:
    """Track last-seen slider values; only report True when something changed."""

    def __init__(self) -> None:
        self._b: Optional[int] = None
        self._c: Optional[int] = None
        self._e: Optional[int] = None

    def has_changed(self, brightness: int, contrast: int, exposure_slider: int) -> bool:
        changed = (self._b != brightness) or (self._c != contrast) or (self._e != exposure_slider)
        if changed:
            self._b, self._c, self._e = brightness, contrast, exposure_slider
        return changed


# ---------------------------------------------------------------------------
# MAIN CONTROL LOOP
# ---------------------------------------------------------------------------

def run_camera(camera_index: int,
               log_path: str,
               snapshot_dir: str,
               width: int,
               height: int,
               attempt_exposure: bool,
               simulate: bool,
               verbose: bool) -> None:
    """Main loop: capture frames, apply params, log changes, support snapshot & simulate modes."""

    # Setup logging file
    init_log(log_path)

    cap = None
    frame_count = 0
    max_simulate_frames = 10

    if not simulate:
        # Initialize hardware camera
        if verbose:
            print(f"[INFO] Opening camera index {camera_index} ...")
        try:
            if platform.system() == "Windows":
                cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(camera_index)
        except Exception as e:
            print(f"[ERROR] Failed to initialize camera: {e}")
            log_params(log_path, camera_index, 0, 0, 0.0, "error", "camera_init_exception")
            return

        if not cap or not cap.isOpened():
            print(f"[ERROR] Cannot open camera index {camera_index}. Try --simulate")
            log_params(log_path, camera_index, 0, 0, 0.0, "error", "failed_open_camera")
            return

        # set capture resolution (best-effort)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
        if verbose:
            print(f"[INFO] Camera opened @ {actual_w}x{actual_h}")

    # Setup UI and trackers
    setup_ui(WINDOW_NAME)
    tracker = ParamTracker()

    # Log session start
    startup_note = f"mode={'simulate' if simulate else 'camera'}"
    log_params(log_path, camera_index, DEFAULT_BRIGHTNESS, DEFAULT_CONTRAST,
               map_exposure_slider_to_value(DEFAULT_EXPOSURE), "start", startup_note)

    print("=" * 60)
    print("CAMERA CONTROL ACTIVE")
    print("Controls: q - quit | s - snapshot | r - reset")
    print("Arrow Up/Down - brightness | Arrow Left/Right - contrast")
    print(f"Logs: {log_path}")
    print(f"Snapshots: {snapshot_dir}")
    print("=" * 60)

    failed_reads = 0
    max_failed_reads = 5

    try:
        while True:
            if simulate:
                frame = create_test_frame(width, height, frame_count)
                ret = True
                frame_count += 1
                if verbose:
                    print(f"[DEBUG] Simulation frame {frame_count}")
                # Auto-snapshot & exit after some frames (helpful for demos)
                if frame_count >= max_simulate_frames:
                    b, c, e_slider = read_trackbars(WINDOW_NAME)
                    e_val = map_exposure_slider_to_value(e_slider)
                    adjusted = apply_brightness_contrast(frame, b, c)
                    snap_fp = save_snapshot(adjusted, snapshot_dir, b, c, e_val)
                    log_params(log_path, camera_index, b, c, e_val, "snapshot", f"auto_sim:{os.path.basename(snap_fp)}")
                    print(f"[INFO] Simulation finished; saved {snap_fp}")
                    break
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    failed_reads += 1
                    if verbose:
                        print(f"[WARN] Failed read #{failed_reads}")
                    if failed_reads >= max_failed_reads:
                        print("[ERROR] Too many consecutive frame read failures, exiting.")
                        log_params(log_path, camera_index, 0, 0, 0.0, "error", f"failed_reads:{failed_reads}")
                        break
                    # short sleep via waitKey below
                    frame = None
                else:
                    failed_reads = 0

            if frame is None:
                # minimal wait to allow keys to be processed
                k = cv2.waitKey(100) & 0xFF
                if k == ord('q'):
                    log_params(log_path, camera_index, DEFAULT_BRIGHTNESS, DEFAULT_CONTRAST,
                               map_exposure_slider_to_value(DEFAULT_EXPOSURE), "exit", "user_quit")
                    break
                continue

            # read sliders
            brightness, contrast, exposure_slider = read_trackbars(WINDOW_NAME)
            exposure_val = map_exposure_slider_to_value(exposure_slider)

            # attempt to set hardware exposure if requested
            if attempt_exposure and not simulate and cap is not None:
                try:
                    cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_val))
                except Exception as e:
                    if verbose:
                        print(f"[WARN] Setting exposure not supported by driver: {e}")

            # apply brightness/contrast transform for preview
            adjusted_frame = apply_brightness_contrast(frame, brightness, contrast)

            # log changes only when detected by tracker
            if tracker.has_changed(brightness, contrast, exposure_slider):
                alpha = contrast / 128.0
                beta = brightness - 128
                note = f"alpha={alpha:.2f},beta={beta},exp_mapped={exposure_val:.3f}"
                log_params(log_path, camera_index, brightness, contrast, exposure_val, "change", note)
                if verbose:
                    print(f"[LOG] change: B={brightness} C={contrast} E={exposure_val:.3f}")

            # overlay a small status text on display
            display = adjusted_frame.copy()
            status_text = f"B:{brightness} C:{contrast} E:{exposure_val:.1f}"
            cv2.putText(display, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                log_params(log_path, camera_index, brightness, contrast, exposure_val, "exit", "user_quit")
                print("[INFO] Quit requested by user.")
                break

            elif key == ord('s'):
                snap_fp = save_snapshot(display, snapshot_dir, brightness, contrast, exposure_val)
                if snap_fp:
                    print(f"[INFO] Snapshot saved: {snap_fp}")
                    log_params(log_path, camera_index, brightness, contrast, exposure_val, "snapshot", os.path.basename(snap_fp))
                else:
                    print("[WARN] Snapshot failed to save.")
            elif key == ord('r'):
                reset_trackbars(WINDOW_NAME)
                reset_exp_val = map_exposure_slider_to_value(DEFAULT_EXPOSURE)
                log_params(log_path, camera_index, DEFAULT_BRIGHTNESS, DEFAULT_CONTRAST, reset_exp_val, "reset", "reset_to_defaults")
                print("[INFO] Trackbars reset to defaults.")

            # Arrow key handling (common codes). Different platforms/webcams may produce different codes.
            elif key == 82:  # Up
                b, c, e = read_trackbars(WINDOW_NAME)
                cv2.setTrackbarPos("Brightness", WINDOW_NAME, min(255, b + 5))
            elif key == 84:  # Down
                b, c, e = read_trackbars(WINDOW_NAME)
                cv2.setTrackbarPos("Brightness", WINDOW_NAME, max(0, b - 5))
            elif key == 83:  # Right
                b, c, e = read_trackbars(WINDOW_NAME)
                cv2.setTrackbarPos("Contrast", WINDOW_NAME, min(255, c + 5))
            elif key == 81:  # Left
                b, c, e = read_trackbars(WINDOW_NAME)
                cv2.setTrackbarPos("Contrast", WINDOW_NAME, max(0, c - 5))

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received; exiting.")
        # log exit with last known values if possible
        try:
            b, c, e_slider = read_trackbars(WINDOW_NAME)
            log_params(log_path, camera_index, b, c, map_exposure_slider_to_value(e_slider), "exit", "keyboard_interrupt")
        except Exception:
            log_params(log_path, camera_index, 0, 0, 0.0, "exit", "keyboard_interrupt_unknown")

    except Exception as ex:
        print(f"[ERROR] Unexpected exception: {ex}")
        traceback.print_exc()
        log_params(log_path, camera_index, 0, 0, 0.0, "error", f"unexpected:{ex}")

    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("[INFO] Resources released. Goodbye.")


# ---------------------------------------------------------------------------
# ARGPARSE + MAIN
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Camera Parameter Automation & Logging Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE_EXAMPLES
    )
    p.add_argument("-c", "--camera", type=int, default=0, help="Camera device index (default 0)")
    p.add_argument("--width", type=int, default=640, help="Capture width (default 640)")
    p.add_argument("--height", type=int, default=480, help="Capture height (default 480)")
    p.add_argument("--log", type=str, default="logs/params_log.csv", help="CSV log path")
    p.add_argument("--save-dir", type=str, default="snapshots", help="Directory for snapshots")
    p.add_argument("--no-exposure-set", action="store_true", help="Do not attempt to set hardware exposure")
    p.add_argument("--simulate", action="store_true", help="Run in simulation mode (no camera required)")
    p.add_argument("--verbose", action="store_true", help="Verbose console output")
    p.add_argument("--preview-log", action="store_true", help="Preview last 5 log entries and exit")
    p.add_argument("--write-readme", action="store_true", help="Write README snippet to file and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.preview_log:
        preview_log(args.log)
        return

    if args.write_readme:
        out_path = "README_camera_control.txt"
        try:
            with open(out_path, "w") as f:
                f.write(README_SNIPPET)
            print(f"[INFO] README written to: {out_path}")
        except Exception as e:
            print(f"[ERROR] Could not write README: {e}")
        return

    # Validate
    if args.width <= 0 or args.height <= 0:
        print("[ERROR] width and height must be positive integers.")
        sys.exit(1)

    run_camera(
        camera_index=args.camera,
        log_path=args.log,
        snapshot_dir=args.save_dir,
        width=args.width,
        height=args.height,
        attempt_exposure=not args.no_exposure_set,
        simulate=args.simulate,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()

