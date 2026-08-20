#!/usr/bin/env python3
"""
thermal_rtsp_analysis.py

Live RTSP thermal analysis for the Engro thermal-camera project.

Given an RTSP stream, the script can:
- On first run, freeze one incoming frame and ask for equipment ROI annotations.
- On first run, ask for thermal scale bar / MAX text / MIN text selections.
- Save those as a reusable preset.
- Continuously estimate the current thermal scale with background OCR.
- Build the current frame's thermal intensity -> temperature LUT.
- Show 3x3 mouse-hover temperature while the stream is live.
- Show per-polygon MIN / AVG / MAX.
- Mark the hottest valid point in every polygon with a red hotspot marker.
- Freeze/unfreeze the display while capture continues in the background.
- Automatically reconnect if RTSP drops.

This script intentionally reuses the project's Step-02 OCR implementation and
Step-03 temperature conversion implementation so recorded-video and RTSP modes
use the same thermal math.

Run from the thermal-plant project root:
    py -3.12 "thermal-stream\\thermal_rtsp_analysis.py" \
        "rtsp://127.0.0.1:8554/thermal91" \
        --preset thermal91

For first-time setup with an explicit Tesseract path:
    py -3.12 "thermal-stream\\thermal_rtsp_analysis.py" \
        "rtsp://127.0.0.1:8554/thermal91" \
        --preset thermal91 \
        --tesseract "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlsplit, urlunsplit

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "OpenCV is required. Install with:\n"
        "  py -3.12 -m pip install opencv-python"
    ) from exc

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "NumPy is required. Install with:\n"
        "  py -3.12 -m pip install numpy"
    ) from exc


# ---------------------------------------------------------------------------
# Import the existing project thermal logic.
# thermal-stream/ sits one directory below the thermal-plant project root.
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import step_02_configure_thermal_scale as t2
    import step_03_process_thermal_temperatures as t3
except ImportError as exc:
    raise SystemExit(
        "Could not import the existing Step-02 / Step-03 files.\n\n"
        "Expected layout:\n"
        "  thermal-plant/\n"
        "    step_02_configure_thermal_scale.py\n"
        "    step_03_process_thermal_temperatures.py\n"
        "    step_code.py\n"
        "    thermal-stream/\n"
        "      thermal_rtsp_analysis.py\n"
    ) from exc


WINDOW_NAME = "Live RTSP Thermal Analysis"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)

MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 820


# =============================================================================
# SMALL UTILITIES
# =============================================================================

def safe_preset_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    value = value.strip("._-")
    return value or "thermal_stream"


def default_preset_name_from_url(url: str) -> str:
    try:
        path = urlsplit(url).path.strip("/")
        if path:
            return safe_preset_name(path.replace("/", "_"))
    except Exception:
        pass
    return "thermal_stream"


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return data


def prompt_yes_no(message: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{message} {suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def build_authenticated_rtsp_url(
    raw_url: str,
    username: Optional[str],
    password: Optional[str],
) -> str:
    raw_url = raw_url.strip()

    if not raw_url.lower().startswith("rtsp://"):
        raise ValueError("Stream URL must start with rtsp://")

    parts = urlsplit(raw_url)

    if "@" in parts.netloc or not username:
        return raw_url

    user_enc = quote(username, safe="")
    pass_enc = quote(password or "", safe="")

    return urlunsplit(
        (
            parts.scheme,
            f"{user_enc}:{pass_enc}@{parts.netloc}",
            parts.path,
            parts.query,
            parts.fragment,
        )
    )


def redact_rtsp_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit(
            (parts.scheme, host, parts.path, parts.query, parts.fragment)
        )
    except Exception:
        return "rtsp://<camera>"


def configure_tesseract(explicit_path: Optional[str]) -> Optional[str]:
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"Tesseract executable not found: {candidate}")
        t2.pytesseract.pytesseract.tesseract_cmd = str(candidate)
        t3.pytesseract.pytesseract.tesseract_cmd = str(candidate)
        return str(candidate)

    if os.name == "nt":
        common = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if common.is_file():
            t2.pytesseract.pytesseract.tesseract_cmd = str(common)
            t3.pytesseract.pytesseract.tesseract_cmd = str(common)
            return str(common)

    return None


def ensure_tesseract_available() -> None:
    try:
        version = t2.pytesseract.get_tesseract_version()
        print(f"Tesseract detected: {version}")
    except Exception as exc:
        raise RuntimeError(
            "Tesseract is required for a live thermal stream because the "
            "stream has no precomputed Step-03 scale_readings.csv.\n"
            'Pass --tesseract "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"'
        ) from exc


def fit_display(frame: np.ndarray) -> Tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = min(
        MAX_DISPLAY_WIDTH / max(1, w),
        MAX_DISPLAY_HEIGHT / max(1, h),
        1.0,
    )

    if scale >= 0.999:
        return frame.copy(), 1.0

    resized = cv2.resize(
        frame,
        (
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def select_roi_scaled(title: str, frame: np.ndarray) -> List[int]:
    display, scale = fit_display(frame)

    roi = cv2.selectROI(
        title,
        display,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(title)

    x, y, w, h = [int(v) for v in roi]

    if w <= 0 or h <= 0:
        raise RuntimeError(f"No valid ROI selected for: {title}")

    if scale <= 0:
        scale = 1.0

    x = int(round(x / scale))
    y = int(round(y / scale))
    w = int(round(w / scale))
    h = int(round(h / scale))

    x = max(0, min(x, frame.shape[1] - 1))
    y = max(0, min(y, frame.shape[0] - 1))
    w = max(1, min(w, frame.shape[1] - x))
    h = max(1, min(h, frame.shape[0] - y))

    return [x, y, w, h]


# =============================================================================
# RTSP CAPTURE: ALWAYS KEEP THE LATEST FRAME
# =============================================================================

class RTSPCapture:
    def __init__(
        self,
        url: str,
        transport: str = "tcp",
        reconnect_delay: float = 2.0,
    ):
        self.url = url
        self.transport = transport
        self.reconnect_delay = max(0.5, float(reconnect_delay))

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._force_reconnect = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_id = 0
        self.latest_frame_time = 0.0

        self.width = 0
        self.height = 0
        self.reported_fps = 0.0

        self.status = "IDLE"
        self.last_error = ""
        self.reconnect_count = 0

        self._receive_times: deque[float] = deque(maxlen=180)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker,
            name="ThermalRTSPCapture",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._force_reconnect.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        self._release()

    def reconnect(self) -> None:
        self._force_reconnect.set()

    def _release(self) -> None:
        cap = self._cap
        self._cap = None

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _open(self) -> bool:
        self.status = "CONNECTING"
        self.last_error = ""

        self._release()

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{self.transport}"
        )

        try:
            cap = cv2.VideoCapture(
                self.url,
                cv2.CAP_FFMPEG,
            )

            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                cap.release()
                self.last_error = "OpenCV could not open the RTSP stream."
                self.status = "RECONNECTING"
                return False

            self._cap = cap
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            self.reported_fps = fps if np.isfinite(fps) else 0.0

            self.status = "CONNECTED"
            return True

        except Exception as exc:
            self.last_error = str(exc)
            self.status = "RECONNECTING"
            self._release()
            return False

    def _worker(self) -> None:
        while not self._stop.is_set():
            if not self._open():
                self._stop.wait(self.reconnect_delay)
                continue

            failures = 0

            while not self._stop.is_set():
                if self._force_reconnect.is_set():
                    self._force_reconnect.clear()
                    self.reconnect_count += 1
                    self.status = "RECONNECTING"
                    break

                cap = self._cap

                if cap is None:
                    break

                ok, frame = cap.read()

                if not ok or frame is None or frame.size == 0:
                    failures += 1

                    if failures >= 3:
                        self.last_error = "RTSP frame read failed."
                        self.reconnect_count += 1
                        self.status = "RECONNECTING"
                        break

                    time.sleep(0.03)
                    continue

                failures = 0
                now = time.monotonic()

                with self._lock:
                    self.latest_frame = frame
                    self.latest_frame_id += 1
                    self.latest_frame_time = now
                    self._receive_times.append(now)

                self.status = "CONNECTED"

            self._release()

            if not self._stop.is_set():
                self._stop.wait(self.reconnect_delay)

        self.status = "STOPPED"

    def get_latest(
        self,
    ) -> Tuple[Optional[np.ndarray], int, float]:
        with self._lock:
            if self.latest_frame is None:
                return None, self.latest_frame_id, self.latest_frame_time

            return (
                self.latest_frame.copy(),
                self.latest_frame_id,
                self.latest_frame_time,
            )

    def measured_fps(self) -> float:
        with self._lock:
            times = list(self._receive_times)

        if len(times) < 2:
            return 0.0

        duration = times[-1] - times[0]

        if duration <= 0:
            return 0.0

        return (len(times) - 1) / duration


def wait_for_first_frame(capture: RTSPCapture) -> np.ndarray:
    print("\nConnecting to RTSP stream...")
    print("The first connection can take several seconds.")

    while True:
        frame, _, _ = capture.get_latest()

        if frame is not None:
            print(
                f"Connected. First frame received: "
                f"{frame.shape[1]} x {frame.shape[0]}"
            )
            return frame

        if capture.last_error:
            print(f"\rStatus: {capture.status} | {capture.last_error[:100]}", end="")

        time.sleep(0.10)


# =============================================================================
# FIRST-TIME EQUIPMENT ANNOTATION
# =============================================================================

class StreamROIAnnotator:
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.display, self.display_scale = fit_display(frame)

        self.current_points: List[Tuple[int, int]] = []
        self.regions: List[dict] = []
        self.mode = "polygon"

        self.window_name = "1/2 - Annotate Equipment ROIs"

    def _display_to_original(self, x: int, y: int) -> Tuple[int, int]:
        scale = self.display_scale if self.display_scale > 0 else 1.0

        ox = int(round(x / scale))
        oy = int(round(y / scale))

        ox = max(0, min(ox, self.frame.shape[1] - 1))
        oy = max(0, min(oy, self.frame.shape[0] - 1))

        return ox, oy

    def _mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        point = self._display_to_original(x, y)

        if self.mode == "polygon":
            self.current_points.append(point)

        elif self.mode == "rectangle":
            if len(self.current_points) < 2:
                self.current_points.append(point)

    @staticmethod
    def _rectangle_points(p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)

        return [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ]

    def _confirm(self):
        if self.mode == "polygon":
            if len(self.current_points) < 3:
                print("Polygon needs at least 3 points.")
                return

            points = [
                [int(x), int(y)]
                for x, y in self.current_points
            ]

        else:
            if len(self.current_points) != 2:
                print("Rectangle needs exactly 2 opposite corners.")
                return

            points = self._rectangle_points(
                self.current_points[0],
                self.current_points[1],
            )

        suggested = f"ROI_{len(self.regions) + 1:02d}"

        print()
        name = input(
            f"Equipment/region name [default: {suggested}]: "
        ).strip()

        if not name:
            name = suggested

        height, width = self.frame.shape[:2]

        normalized_points = [
            [
                float(x) / float(width),
                float(y) / float(height),
            ]
            for x, y in points
        ]

        self.regions.append(
            {
                "id": len(self.regions) + 1,
                "name": name,
                "shape_type": self.mode,
                "pixel_points": points,
                "normalized_points": normalized_points,
            }
        )

        print(f"Confirmed: {name}")
        self.current_points = []

    def _draw(self) -> np.ndarray:
        output = self.frame.copy()

        for region in self.regions:
            points = np.array(
                region["pixel_points"],
                dtype=np.int32,
            )

            cv2.polylines(
                output,
                [points],
                True,
                RED,
                2,
            )

            min_x = int(np.min(points[:, 0]))
            min_y = int(np.min(points[:, 1]))

            cv2.putText(
                output,
                region["name"],
                (min_x, max(25, min_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                RED,
                2,
                cv2.LINE_AA,
            )

        if self.current_points:
            for point in self.current_points:
                cv2.circle(
                    output,
                    point,
                    5,
                    YELLOW,
                    -1,
                )

            if self.mode == "polygon" and len(self.current_points) >= 2:
                cv2.polylines(
                    output,
                    [
                        np.array(
                            self.current_points,
                            dtype=np.int32,
                        )
                    ],
                    False,
                    YELLOW,
                    2,
                )

            elif self.mode == "rectangle" and len(self.current_points) == 2:
                cv2.rectangle(
                    output,
                    self.current_points[0],
                    self.current_points[1],
                    YELLOW,
                    2,
                )

        instructions = (
            f"MODE {self.mode.upper()} | Click points | ENTER confirm | "
            "P polygon | R rectangle | BACKSPACE undo point | "
            "U undo region | C clear | S finish"
        )

        cv2.rectangle(
            output,
            (0, 0),
            (output.shape[1], 36),
            BLACK,
            -1,
        )

        cv2.putText(
            output,
            instructions,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            WHITE,
            1,
            cv2.LINE_AA,
        )

        if self.display_scale != 1.0:
            output = cv2.resize(
                output,
                (
                    self.display.shape[1],
                    self.display.shape[0],
                ),
                interpolation=cv2.INTER_AREA,
            )

        return output

    def run(self) -> List[dict]:
        print("\n============================================================")
        print("FIRST-TIME STREAM SETUP: EQUIPMENT ANNOTATIONS")
        print("============================================================")
        print("Annotate the equipment polygons/rectangles on this frozen frame.")
        print("Controls:")
        print("  Click      add point")
        print("  ENTER      confirm current region")
        print("  P          polygon mode")
        print("  R          rectangle mode")
        print("  BACKSPACE  remove last point")
        print("  U          remove last confirmed region")
        print("  C          clear unfinished region")
        print("  S          finish ROI setup")

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse)

        while True:
            cv2.imshow(self.window_name, self._draw())

            key = cv2.waitKey(20) & 0xFF

            if key in (10, 13):
                self._confirm()

            elif key == 8:
                if self.current_points:
                    self.current_points.pop()

            elif key in (ord("p"), ord("P")):
                self.current_points = []
                self.mode = "polygon"
                print("Switched to POLYGON.")

            elif key in (ord("r"), ord("R")):
                self.current_points = []
                self.mode = "rectangle"
                print("Switched to RECTANGLE.")

            elif key in (ord("u"), ord("U")):
                if self.regions:
                    removed = self.regions.pop()
                    print(f"Removed region: {removed['name']}")

            elif key in (ord("c"), ord("C")):
                self.current_points = []

            elif key in (ord("s"), ord("S")):
                if not self.regions:
                    print("Annotate at least one region first.")
                    continue
                break

        cv2.destroyWindow(self.window_name)
        cv2.destroyAllWindows()

        return self.regions


# =============================================================================
# FIRST-TIME THERMAL SCALE CONFIGURATION
# =============================================================================

def configure_scale_from_frame(
    frame: np.ndarray,
    preset_dir: Path,
    args,
) -> dict:
    ensure_tesseract_available()

    print("\n============================================================")
    print("FIRST-TIME STREAM SETUP: THERMAL SCALE")
    print("============================================================")
    print("Select three rectangles from the SAME frozen frame:")
    print("  1) vertical thermal bar only")
    print("  2) MAX temperature text")
    print("  3) MIN temperature text")

    while True:
        print("\n1/3 Select THERMAL BAR")
        bar_roi = select_roi_scaled(
            "1/3 - Select THERMAL BAR",
            frame,
        )

        print("2/3 Select MAX temperature text")
        max_roi = select_roi_scaled(
            "2/3 - Select MAX temperature text",
            frame,
        )

        print("3/3 Select MIN temperature text")
        min_roi = select_roi_scaled(
            "3/3 - Select MIN temperature text",
            frame,
        )

        min_candidates = []
        max_candidates = []

        for method, psm in t2.OCR_METHODS_TO_TEST:
            min_candidates.append(
                t2.run_one_ocr(
                    frame,
                    min_roi,
                    method=method,
                    psm=psm,
                    decimal_places=int(args.decimal_places),
                    padding=int(args.ocr_padding),
                )
            )

            max_candidates.append(
                t2.run_one_ocr(
                    frame,
                    max_roi,
                    method=method,
                    psm=psm,
                    decimal_places=int(args.decimal_places),
                    padding=int(args.ocr_padding),
                )
            )

        best_pair = t2.choose_best_pair(
            min_candidates=min_candidates,
            max_candidates=max_candidates,
            min_allowed=float(args.min_allowed_temp),
            max_allowed=float(args.max_allowed_temp),
            min_scale_span=float(args.min_scale_span),
        )

        print("\nOCR TEST RESULTS")
        print("-" * 78)

        for min_candidate, max_candidate in zip(
            min_candidates,
            max_candidates,
        ):
            min_value = (
                "FAIL"
                if not np.isfinite(min_candidate["value"])
                else f"{min_candidate['value']:.1f}"
            )

            max_value = (
                "FAIL"
                if not np.isfinite(max_candidate["value"])
                else f"{max_candidate['value']:.1f}"
            )

            print(
                f"{min_candidate['method']:24} "
                f"PSM{min_candidate['psm']} | "
                f"MIN {min_candidate['raw_text']!r:>10} -> {min_value:>6} | "
                f"MAX {max_candidate['raw_text']!r:>10} -> {max_value:>6}"
            )

        if best_pair is None:
            print("\nNo reliable MIN/MAX pair was found.")

            if prompt_yes_no(
                "Redraw the scale regions?",
                default=True,
            ):
                continue

            raise RuntimeError(
                "Thermal scale setup cancelled."
            )

        best_min, best_max = best_pair

        print("\nSelected scale:")
        print(
            f"  MIN {best_min['value']:.1f} C "
            f"using {best_min['method']}/PSM{best_min['psm']}"
        )
        print(
            f"  MAX {best_max['value']:.1f} C "
            f"using {best_max['method']}/PSM{best_max['psm']}"
        )

        if not prompt_yes_no(
            "Use this thermal scale setup?",
            default=True,
        ):
            continue

        config = {
            "video": {
                "source_type": "rtsp",
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            },
            "scale": {
                "bar_roi": bar_roi,
                "max_text_roi": max_roi,
                "min_text_roi": min_roi,
                "bar_horizontal_inner_fraction": 0.20,
                "display_decimal_places": int(args.decimal_places),
                "ocr_padding_pixels": int(args.ocr_padding),
                "min_ocr_method": best_min["method"],
                "min_ocr_psm": int(best_min["psm"]),
                "max_ocr_method": best_max["method"],
                "max_ocr_psm": int(best_max["psm"]),
                "reference_ocr_min_temp": float(best_min["value"]),
                "reference_ocr_max_temp": float(best_max["value"]),
                "reference_ocr_min_text": best_min["raw_text"],
                "reference_ocr_max_text": best_max["raw_text"],
                "min_scale_span_c": float(args.min_scale_span),
            },
        }

        preview = frame.copy()

        t2.draw_box(
            preview,
            bar_roi,
            "THERMAL BAR",
            CYAN,
        )

        t2.draw_box(
            preview,
            max_roi,
            f"MAX {best_max['value']:.1f} C",
            GREEN,
        )

        t2.draw_box(
            preview,
            min_roi,
            f"MIN {best_min['value']:.1f} C",
            YELLOW,
        )

        cv2.imwrite(
            str(preset_dir / "scale_config_preview.png"),
            preview,
        )

        return config


def build_or_load_preset(
    first_frame: np.ndarray,
    preset_dir: Path,
    stream_url_redacted: str,
    args,
) -> Tuple[dict, dict]:
    regions_path = preset_dir / "regions.json"
    scale_path = preset_dir / "scale_config.json"

    existing_complete = (
        regions_path.is_file()
        and scale_path.is_file()
    )

    if existing_complete and not args.reconfigure:
        print("\nReusable stream preset found:")
        print(f"  Regions: {regions_path}")
        print(f"  Scale  : {scale_path}")

        if prompt_yes_no(
            "Reuse this preset?",
            default=True,
        ):
            regions_data = load_json(regions_path)
            scale_config = load_json(scale_path)
            return regions_data, scale_config

    preset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    regions = StreamROIAnnotator(
        first_frame
    ).run()

    regions_data = {
        "video": {
            "source_type": "rtsp",
            "source": stream_url_redacted,
            "width": int(first_frame.shape[1]),
            "height": int(first_frame.shape[0]),
        },
        "regions": regions,
    }

    save_json(
        regions_path,
        regions_data,
    )

    cv2.imwrite(
        str(preset_dir / "reference_frame.png"),
        first_frame,
    )

    annotated_preview = first_frame.copy()

    for region in regions:
        points = np.array(
            region["pixel_points"],
            dtype=np.int32,
        )

        cv2.polylines(
            annotated_preview,
            [points],
            True,
            RED,
            2,
        )

    cv2.imwrite(
        str(preset_dir / "regions_preview.png"),
        annotated_preview,
    )

    scale_config = configure_scale_from_frame(
        first_frame,
        preset_dir,
        args,
    )

    save_json(
        scale_path,
        scale_config,
    )

    print("\nStream preset saved:")
    print(f"  {preset_dir}")

    return regions_data, scale_config


# =============================================================================
# LIVE OCR WITH CONFIRMED LARGE-JUMP HANDLING
# =============================================================================

class LiveScaleTracker:
    def __init__(
        self,
        scale: dict,
        args,
    ):
        self.scale = scale
        self.args = args

        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="live-thermal-ocr",
        )

        self.future = None
        self.future_submitted_at = None

        self.interval_seconds = (
            1.0 / float(args.ocr_hz)
        )

        self.last_submit_time = -(10**9)

        self.trusted_min = self._finite_or_none(
            scale.get("reference_ocr_min_temp")
        )

        self.trusted_max = self._finite_or_none(
            scale.get("reference_ocr_max_temp")
        )

        self.pending_jump: Optional[Tuple[float, float]] = None

        self.status = (
            "REFERENCE SCALE"
            if (
                self.trusted_min is not None
                and self.trusted_max is not None
            )
            else "WAITING FOR OCR"
        )

    @staticmethod
    def _finite_or_none(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    def close(self):
        self.executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

    def _worker(self, frame: np.ndarray):
        scale = self.scale

        min_value, min_text, _ = t3.ocr_temperature(
            frame,
            scale["min_text_roi"],
            decimal_places=int(
                scale.get(
                    "display_decimal_places",
                    1,
                )
            ),
            method=scale.get(
                "min_ocr_method",
                "green_difference",
            ),
            psm=int(
                scale.get(
                    "min_ocr_psm",
                    7,
                )
            ),
            padding=int(
                scale.get(
                    "ocr_padding_pixels",
                    6,
                )
            ),
        )

        max_value, max_text, _ = t3.ocr_temperature(
            frame,
            scale["max_text_roi"],
            decimal_places=int(
                scale.get(
                    "display_decimal_places",
                    1,
                )
            ),
            method=scale.get(
                "max_ocr_method",
                "green_difference",
            ),
            psm=int(
                scale.get(
                    "max_ocr_psm",
                    7,
                )
            ),
            padding=int(
                scale.get(
                    "ocr_padding_pixels",
                    6,
                )
            ),
        )

        if t3._basic_scale_pair_valid(
            min_value=min_value,
            max_value=max_value,
            min_allowed=float(
                self.args.min_allowed_temp
            ),
            max_allowed=float(
                self.args.max_allowed_temp
            ),
            min_scale_span=float(
                self.args.min_scale_span
            ),
        ):
            return {
                "min_temp": float(min_value),
                "max_temp": float(max_value),
                "min_text": min_text,
                "max_text": max_text,
                "source": "PRIMARY",
            }

        min_candidates = t3._collect_ocr_candidates(
            frame=frame,
            roi=scale["min_text_roi"],
            decimal_places=int(
                scale.get(
                    "display_decimal_places",
                    1,
                )
            ),
            primary_method=scale.get(
                "min_ocr_method",
                "green_difference",
            ),
            primary_psm=int(
                scale.get(
                    "min_ocr_psm",
                    7,
                )
            ),
            padding=int(
                scale.get(
                    "ocr_padding_pixels",
                    6,
                )
            ),
        )

        max_candidates = t3._collect_ocr_candidates(
            frame=frame,
            roi=scale["max_text_roi"],
            decimal_places=int(
                scale.get(
                    "display_decimal_places",
                    1,
                )
            ),
            primary_method=scale.get(
                "max_ocr_method",
                "green_difference",
            ),
            primary_psm=int(
                scale.get(
                    "max_ocr_psm",
                    7,
                )
            ),
            padding=int(
                scale.get(
                    "ocr_padding_pixels",
                    6,
                )
            ),
        )

        # For a live stream we first choose by physical validity and OCR
        # appearance. Temporal validation is performed below so large legitimate
        # jumps can be accepted only after confirmation.
        best_pair = t2.choose_best_pair(
            min_candidates=[
                {
                    "value": c["value"],
                    "raw_text": c["text"],
                    "method": c["method"],
                    "psm": c["psm"],
                }
                for c in min_candidates
            ],
            max_candidates=[
                {
                    "value": c["value"],
                    "raw_text": c["text"],
                    "method": c["method"],
                    "psm": c["psm"],
                }
                for c in max_candidates
            ],
            min_allowed=float(
                self.args.min_allowed_temp
            ),
            max_allowed=float(
                self.args.max_allowed_temp
            ),
            min_scale_span=float(
                self.args.min_scale_span
            ),
        )

        if best_pair is None:
            return None

        best_min, best_max = best_pair

        return {
            "min_temp": float(best_min["value"]),
            "max_temp": float(best_max["value"]),
            "min_text": best_min["raw_text"],
            "max_text": best_max["raw_text"],
            "source": "RETRY",
        }

    def _accept_result(
        self,
        result: Optional[dict],
    ):
        if result is None:
            self.status = "OCR FAILED - HOLD PREVIOUS"
            self.pending_jump = None
            return

        candidate_min = float(
            result["min_temp"]
        )

        candidate_max = float(
            result["max_temp"]
        )

        if (
            self.trusted_min is None
            or self.trusted_max is None
        ):
            self.trusted_min = candidate_min
            self.trusted_max = candidate_max
            self.pending_jump = None
            self.status = "LIVE OCR"
            return

        max_jump = float(
            self.args.max_scale_jump
        )

        normal_change = (
            max_jump <= 0
            or (
                abs(
                    candidate_min
                    - self.trusted_min
                )
                <= max_jump
                and
                abs(
                    candidate_max
                    - self.trusted_max
                )
                <= max_jump
            )
        )

        if normal_change:
            self.trusted_min = candidate_min
            self.trusted_max = candidate_max
            self.pending_jump = None
            self.status = "LIVE OCR"
            return

        # Large jump: do NOT trust one OCR sample.
        # Require the next OCR sample to agree with this new candidate.
        tolerance = float(
            self.args.jump_confirmation_tolerance
        )

        if self.pending_jump is not None:
            pending_min, pending_max = (
                self.pending_jump
            )

            agrees = (
                abs(
                    candidate_min
                    - pending_min
                )
                <= tolerance
                and
                abs(
                    candidate_max
                    - pending_max
                )
                <= tolerance
            )

            if agrees:
                self.trusted_min = candidate_min
                self.trusted_max = candidate_max
                self.pending_jump = None
                self.status = "LIVE OCR - CONFIRMED SCALE JUMP"
                return

        self.pending_jump = (
            candidate_min,
            candidate_max,
        )

        self.status = (
            "POSSIBLE SCALE JUMP - WAITING FOR CONFIRMATION"
        )

    def poll(self):
        if (
            self.future is None
            or not self.future.done()
        ):
            return

        future = self.future
        self.future = None

        try:
            result = future.result()
        except Exception as exc:
            self.status = (
                f"OCR ERROR - HOLD PREVIOUS: {str(exc)[:60]}"
            )
            return

        self._accept_result(
            result
        )

    def maybe_submit(
        self,
        frame: np.ndarray,
    ):
        self.poll()

        if self.future is not None:
            return

        now = time.monotonic()

        if (
            now
            - self.last_submit_time
            < self.interval_seconds
        ):
            return

        self.last_submit_time = now
        self.future_submitted_at = now

        self.future = self.executor.submit(
            self._worker,
            frame.copy(),
        )

    def current(
        self,
    ) -> Optional[Tuple[float, float]]:
        self.poll()

        if (
            self.trusted_min is None
            or self.trusted_max is None
        ):
            return None

        return (
            float(self.trusted_min),
            float(self.trusted_max),
        )


# =============================================================================
# REGION ANALYSIS
# =============================================================================

class RegionAnalyzer:
    def __init__(
        self,
        regions: Sequence[dict],
        width: int,
        height: int,
        erode_pixels: int = 0,
    ):
        self.regions = list(regions)
        self.width = int(width)
        self.height = int(height)

        self.flat_indices: Dict[int, np.ndarray] = {}

        for region in self.regions:
            mask = t3.make_polygon_mask(
                (height, width),
                region["pixel_points"],
                erode_pixels=max(
                    0,
                    int(erode_pixels),
                ),
            )

            self.flat_indices[
                int(region["id"])
            ] = np.flatnonzero(
                mask.reshape(-1) > 0
            )

    def analyze(
        self,
        temperature_map: np.ndarray,
        valid_pixel_mask: np.ndarray,
    ) -> Dict[int, Optional[dict]]:
        temp_flat = temperature_map.reshape(
            -1
        )

        valid_flat = valid_pixel_mask.reshape(
            -1
        )

        results = {}

        for region in self.regions:
            region_id = int(
                region["id"]
            )

            indices = self.flat_indices[
                region_id
            ]

            if indices.size == 0:
                results[region_id] = None
                continue

            good = valid_flat[
                indices
            ]

            if not np.any(good):
                results[region_id] = None
                continue

            valid_indices = indices[
                good
            ]

            values = temp_flat[
                valid_indices
            ]

            finite = np.isfinite(
                values
            )

            if not np.any(finite):
                results[region_id] = None
                continue

            values = values[
                finite
            ]

            valid_indices = valid_indices[
                finite
            ]

            local_max_idx = int(
                np.argmax(values)
            )

            max_flat_idx = int(
                valid_indices[
                    local_max_idx
                ]
            )

            max_y, max_x = divmod(
                max_flat_idx,
                self.width,
            )

            results[region_id] = {
                "min_temp": float(
                    np.min(values)
                ),
                "avg_temp": float(
                    np.mean(values)
                ),
                "max_temp": float(
                    values[
                        local_max_idx
                    ]
                ),
                "max_location": (
                    int(max_x),
                    int(max_y),
                ),
                "pixel_count": int(
                    values.size
                ),
            }

        return results


def point_inside_roi(
    x: int,
    y: int,
    roi: Sequence[int],
) -> bool:
    rx, ry, rw, rh = [
        int(v)
        for v in roi
    ]

    return (
        rx <= x < rx + rw
        and ry <= y < ry + rh
    )


def sample_hover_3x3(
    frame: np.ndarray,
    temperature_map: np.ndarray,
    valid_pixel_mask: np.ndarray,
    x: int,
    y: int,
    scale: dict,
) -> dict:
    height, width = frame.shape[:2]

    if not (
        0 <= x < width
        and 0 <= y < height
    ):
        return {
            "valid": False,
            "reason": "OUTSIDE FRAME",
        }

    for roi in (
        scale["bar_roi"],
        scale["max_text_roi"],
        scale["min_text_roi"],
    ):
        if point_inside_roi(
            x,
            y,
            roi,
        ):
            return {
                "valid": False,
                "reason": "THERMAL SCALE / UI",
            }

    x1 = max(0, x - 1)
    x2 = min(width, x + 2)
    y1 = max(0, y - 1)
    y2 = min(height, y + 2)

    patch_temp = temperature_map[
        y1:y2,
        x1:x2,
    ]

    patch_valid = valid_pixel_mask[
        y1:y2,
        x1:x2,
    ]

    values = patch_temp[
        patch_valid
        & np.isfinite(
            patch_temp
        )
    ]

    if values.size == 0:
        return {
            "valid": False,
            "reason": "NO VALID THERMAL PIXELS",
        }

    return {
        "valid": True,
        "temperature": float(
            np.mean(values)
        ),
        "minimum": float(
            np.min(values)
        ),
        "maximum": float(
            np.max(values)
        ),
        "sample_count": int(
            values.size
        ),
    }


# =============================================================================
# DRAWING HELPERS
# =============================================================================

def draw_text_box(
    image: np.ndarray,
    lines: Sequence[str],
    origin: Tuple[int, int],
    border_color=WHITE,
    font_scale: float = 0.47,
):
    if not lines:
        return

    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    line_height = 20

    widths = [
        cv2.getTextSize(
            str(line),
            font,
            font_scale,
            thickness,
        )[0][0]
        for line in lines
    ]

    box_w = max(
        widths,
        default=100,
    ) + 16

    box_h = (
        len(lines)
        * line_height
        + 10
    )

    x = max(
        0,
        min(
            x,
            image.shape[1]
            - box_w
            - 1,
        ),
    )

    y = max(
        box_h,
        min(
            y,
            image.shape[0] - 1,
        ),
    )

    x2 = min(
        image.shape[1] - 1,
        x + box_w,
    )

    y1 = max(
        0,
        y - box_h,
    )

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (x, y1),
        (x2, y),
        BLACK,
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.72,
        image,
        0.28,
        0,
        image,
    )

    cv2.rectangle(
        image,
        (x, y1),
        (x2, y),
        border_color,
        1,
    )

    text_y = y1 + 18

    for line in lines:
        cv2.putText(
            image,
            str(line),
            (x + 8, text_y),
            font,
            font_scale,
            WHITE,
            thickness,
            cv2.LINE_AA,
        )

        text_y += line_height


def draw_region_overlay(
    display: np.ndarray,
    region: dict,
    stats: Optional[dict],
    scale_x: float,
    scale_y: float,
):
    points = np.array(
        region["pixel_points"],
        dtype=np.float32,
    )

    display_points = points.copy()
    display_points[:, 0] *= float(
        scale_x
    )
    display_points[:, 1] *= float(
        scale_y
    )

    display_points = np.round(
        display_points
    ).astype(np.int32)

    cv2.polylines(
        display,
        [display_points],
        True,
        RED,
        2,
    )

    min_x = int(
        np.min(
            display_points[:, 0]
        )
    )

    min_y = int(
        np.min(
            display_points[:, 1]
        )
    )

    if stats is None:
        lines = [
            str(region["name"]),
            "NO VALID THERMAL PIXELS",
        ]

    else:
        lines = [
            str(region["name"]),
            (
                f"Min {stats['min_temp']:.1f} C | "
                f"Avg {stats['avg_temp']:.1f} C | "
                f"Max {stats['max_temp']:.1f} C"
            ),
        ]

    draw_text_box(
        display,
        lines,
        (
            max(0, min_x),
            max(44, min_y - 5),
        ),
        border_color=RED,
        font_scale=0.43,
    )

    if stats is None:
        return

    max_x, max_y = stats[
        "max_location"
    ]

    draw_x = int(
        round(
            max_x
            * scale_x
        )
    )

    draw_y = int(
        round(
            max_y
            * scale_y
        )
    )

    cv2.circle(
        display,
        (draw_x, draw_y),
        6,
        RED,
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        display,
        (draw_x, draw_y),
        9,
        WHITE,
        1,
        cv2.LINE_AA,
    )

    label = (
        f"{stats['max_temp']:.1f} C"
    )

    tx = max(
        0,
        min(
            display.shape[1] - 90,
            draw_x + 12,
        ),
    )

    ty = max(
        18,
        draw_y - 8,
    )

    cv2.putText(
        display,
        label,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        BLACK,
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        label,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        RED,
        1,
        cv2.LINE_AA,
    )


def draw_hover_tooltip(
    image: np.ndarray,
    x: int,
    y: int,
    lines: Sequence[str],
    valid: bool,
):
    color = (
        GREEN
        if valid
        else RED
    )

    cv2.drawMarker(
        image,
        (x, y),
        color,
        cv2.MARKER_CROSS,
        20,
        2,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    thickness = 1
    line_height = 20

    widths = [
        cv2.getTextSize(
            line,
            font,
            font_scale,
            thickness,
        )[0][0]
        for line in lines
    ]

    box_w = max(
        widths,
        default=100,
    ) + 18

    box_h = (
        len(lines)
        * line_height
        + 10
    )

    x1 = x + 15
    y1 = y + 15

    if (
        x1 + box_w
        >= image.shape[1]
    ):
        x1 = max(
            0,
            x - box_w - 15,
        )

    if (
        y1 + box_h
        >= image.shape[0]
    ):
        y1 = max(
            0,
            y - box_h - 15,
        )

    x2 = min(
        image.shape[1] - 1,
        x1 + box_w,
    )

    y2 = min(
        image.shape[0] - 1,
        y1 + box_h,
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        BLACK,
        -1,
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        1,
    )

    text_y = y1 + 18

    for line in lines:
        cv2.putText(
            image,
            line,
            (
                x1 + 8,
                text_y,
            ),
            font,
            font_scale,
            WHITE,
            thickness,
            cv2.LINE_AA,
        )

        text_y += line_height


# =============================================================================
# LIVE ANALYSIS VIEWER
# =============================================================================

class LiveThermalViewer:
    def __init__(
        self,
        capture: RTSPCapture,
        regions_data: dict,
        scale_config: dict,
        preset_dir: Path,
        args,
    ):
        self.capture = capture
        self.regions_data = regions_data
        self.regions = list(
            regions_data["regions"]
        )
        self.scale_config = scale_config
        self.scale = scale_config[
            "scale"
        ]
        self.preset_dir = preset_dir
        self.args = args

        first_frame, first_id, first_time = (
            capture.get_latest()
        )

        if first_frame is None:
            raise RuntimeError(
                "No RTSP frame is available."
            )

        self.current_frame = first_frame
        self.current_frame_id = first_id
        self.current_frame_time = first_time

        self.height, self.width = (
            first_frame.shape[:2]
        )

        self._validate_preset_resolution()

        self.region_analyzer = (
            RegionAnalyzer(
                self.regions,
                self.width,
                self.height,
                erode_pixels=int(
                    args.erode
                ),
            )
        )

        self.scale_tracker = (
            LiveScaleTracker(
                self.scale,
                args,
            )
        )

        self.temperature_map = None
        self.valid_pixel_mask = None
        self.region_stats = {}
        self.current_min = None
        self.current_max = None

        self.frozen = False
        self.frozen_frame_id = None

        self.mouse_display_x = None
        self.mouse_display_y = None
        self.display_width = None
        self.display_height = None

        self.screenshot_dir = (
            Path(
                args.screenshot_dir
            )
            .expanduser()
            .resolve()
        )

        self.last_processed_id = -1

        self._process_current_frame()

    def _validate_preset_resolution(self):
        region_video = (
            self.regions_data
            .get(
                "video",
                {},
            )
        )

        rw = int(
            region_video.get(
                "width",
                self.width,
            )
        )

        rh = int(
            region_video.get(
                "height",
                self.height,
            )
        )

        if (
            rw,
            rh,
        ) != (
            self.width,
            self.height,
        ):
            raise RuntimeError(
                "Saved ROI preset resolution does not match the RTSP stream.\n"
                f"Preset: {rw} x {rh}\n"
                f"Stream: {self.width} x {self.height}\n"
                "Run again with --reconfigure."
            )

        cfg_video = (
            self.scale_config
            .get(
                "video",
                {},
            )
        )

        sw = int(
            cfg_video.get(
                "width",
                self.width,
            )
        )

        sh = int(
            cfg_video.get(
                "height",
                self.height,
            )
        )

        if (
            sw,
            sh,
        ) != (
            self.width,
            self.height,
        ):
            raise RuntimeError(
                "Saved thermal-scale preset resolution does not match the stream.\n"
                f"Preset: {sw} x {sh}\n"
                f"Stream: {self.width} x {self.height}\n"
                "Run again with --reconfigure."
            )

    def close(self):
        self.scale_tracker.close()
        cv2.destroyAllWindows()

    def _process_current_frame(self):
        self.scale_tracker.maybe_submit(
            self.current_frame
        )

        scale_pair = (
            self.scale_tracker.current()
        )

        if scale_pair is None:
            self.temperature_map = None
            self.valid_pixel_mask = None
            self.region_stats = {}
            return

        (
            self.current_min,
            self.current_max,
        ) = scale_pair

        lut = (
            t3.build_grayscale_temperature_lut(
                self.current_frame,
                bar_roi=self.scale[
                    "bar_roi"
                ],
                min_temp=self.current_min,
                max_temp=self.current_max,
                inner_fraction=float(
                    self.scale.get(
                        "bar_horizontal_inner_fraction",
                        0.20,
                    )
                ),
            )
        )

        gray = cv2.cvtColor(
            self.current_frame,
            cv2.COLOR_BGR2GRAY,
        )

        self.temperature_map = (
            lut[
                gray
            ]
        )

        self.valid_pixel_mask = (
            t3.neutral_color_mask(
                self.current_frame,
                max_channel_difference=int(
                    self.args.max_channel_difference
                ),
            )
        )

        self.region_stats = (
            self.region_analyzer.analyze(
                self.temperature_map,
                self.valid_pixel_mask,
            )
        )

    def _update_live_frame(self):
        if self.frozen:
            self.scale_tracker.poll()
            return

        frame, frame_id, frame_time = (
            self.capture.get_latest()
        )

        if (
            frame is None
            or frame_id
            == self.last_processed_id
        ):
            self.scale_tracker.poll()
            return

        if (
            frame.shape[1]
            != self.width
            or frame.shape[0]
            != self.height
        ):
            raise RuntimeError(
                "RTSP stream resolution changed while running. "
                "Reconfigure the preset for the new resolution."
            )

        self.current_frame = frame
        self.current_frame_id = frame_id
        self.current_frame_time = frame_time
        self.last_processed_id = frame_id

        self._process_current_frame()

    def _display_size(self):
        scale = min(
            MAX_DISPLAY_WIDTH
            / self.width,
            MAX_DISPLAY_HEIGHT
            / self.height,
            1.0,
        )

        return (
            max(
                1,
                int(
                    round(
                        self.width
                        * scale
                    )
                ),
            ),
            max(
                1,
                int(
                    round(
                        self.height
                        * scale
                    )
                ),
            ),
        )

    def _mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_MOUSEMOVE:
            return

        self.mouse_display_x = int(x)
        self.mouse_display_y = int(y)

    def _mouse_original(
        self,
    ) -> Optional[Tuple[int, int]]:
        if (
            self.mouse_display_x
            is None
            or self.mouse_display_y
            is None
            or self.display_width
            is None
            or self.display_height
            is None
        ):
            return None

        if not (
            0
            <= self.mouse_display_x
            < self.display_width
            and 0
            <= self.mouse_display_y
            < self.display_height
        ):
            return None

        x = int(
            round(
                self.mouse_display_x
                * self.width
                / self.display_width
            )
        )

        y = int(
            round(
                self.mouse_display_y
                * self.height
                / self.display_height
            )
        )

        return (
            max(
                0,
                min(
                    x,
                    self.width - 1,
                ),
            ),
            max(
                0,
                min(
                    y,
                    self.height - 1,
                ),
            ),
        )

    def _render(self) -> np.ndarray:
        (
            self.display_width,
            self.display_height,
        ) = self._display_size()

        display = cv2.resize(
            self.current_frame,
            (
                self.display_width,
                self.display_height,
            ),
            interpolation=(
                cv2.INTER_AREA
                if (
                    self.display_width
                    < self.width
                )
                else cv2.INTER_LINEAR
            ),
        )

        scale_x = (
            self.display_width
            / self.width
        )

        scale_y = (
            self.display_height
            / self.height
        )

        if self.temperature_map is not None:
            for region in self.regions:
                draw_region_overlay(
                    display,
                    region,
                    self.region_stats.get(
                        int(
                            region["id"]
                        )
                    ),
                    scale_x,
                    scale_y,
                )

        mode = (
            "FROZEN"
            if self.frozen
            else "LIVE"
        )

        frame_age = (
            max(
                0.0,
                time.monotonic()
                - self.current_frame_time,
            )
            if self.current_frame_time
            else 0.0
        )

        status_lines = [
            (
                f"{mode} | "
                f"{self.capture.status} | "
                "HOVER + ROI STATS + HOTSPOTS"
            ),
            (
                f"Receive FPS {self.capture.measured_fps():.1f} | "
                f"Frame age {frame_age * 1000:.0f} ms | "
                f"Reconnects {self.capture.reconnect_count}"
            ),
        ]

        if (
            self.current_min is not None
            and self.current_max is not None
        ):
            status_lines.append(
                (
                    f"Scale "
                    f"{self.current_min:.1f}.."
                    f"{self.current_max:.1f} C | "
                    f"{self.scale_tracker.status}"
                )
            )

        else:
            status_lines.append(
                f"Scale unavailable | {self.scale_tracker.status}"
            )

        status_lines.append(
            f"Regions: {len(self.regions)}"
        )

        draw_text_box(
            display,
            status_lines,
            (10, 92),
            border_color=YELLOW,
            font_scale=0.46,
        )

        mouse = self._mouse_original()

        if mouse is not None:
            ox, oy = mouse

            if (
                self.temperature_map
                is None
                or self.valid_pixel_mask
                is None
            ):
                draw_hover_tooltip(
                    display,
                    self.mouse_display_x,
                    self.mouse_display_y,
                    [
                        f"Pixel x={ox}, y={oy}",
                        "Temperature unavailable",
                    ],
                    valid=False,
                )

            else:
                result = sample_hover_3x3(
                    self.current_frame,
                    self.temperature_map,
                    self.valid_pixel_mask,
                    ox,
                    oy,
                    self.scale,
                )

                if result["valid"]:
                    draw_hover_tooltip(
                        display,
                        self.mouse_display_x,
                        self.mouse_display_y,
                        [
                            (
                                f"3x3 avg: "
                                f"{result['temperature']:.1f} C"
                            ),
                            (
                                f"Range: "
                                f"{result['minimum']:.1f}.."
                                f"{result['maximum']:.1f} C"
                            ),
                            f"Pixel x={ox}, y={oy}",
                            (
                                f"Valid: "
                                f"{result['sample_count']}/9"
                            ),
                        ],
                        valid=True,
                    )

                else:
                    draw_hover_tooltip(
                        display,
                        self.mouse_display_x,
                        self.mouse_display_y,
                        [
                            f"Pixel x={ox}, y={oy}",
                            result["reason"],
                            "Temperature: N/A",
                        ],
                        valid=False,
                    )

        controls = (
            "SPACE Freeze/Live | S Screenshot | "
            "R Reconnect | Q/ESC Quit"
        )

        cv2.rectangle(
            display,
            (
                0,
                display.shape[0] - 30,
            ),
            (
                display.shape[1],
                display.shape[0],
            ),
            BLACK,
            -1,
        )

        cv2.putText(
            display,
            controls,
            (
                10,
                display.shape[0] - 9,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            WHITE,
            1,
            cv2.LINE_AA,
        )

        return display

    def _save_screenshot(
        self,
        display: np.ndarray,
    ):
        self.screenshot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        path = (
            self.screenshot_dir
            / f"thermal_stream_{timestamp}.png"
        )

        if cv2.imwrite(
            str(path),
            display,
        ):
            print(
                f"\nScreenshot saved: {path}"
            )

    def run(self):
        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_AUTOSIZE,
        )

        cv2.setMouseCallback(
            WINDOW_NAME,
            self._mouse,
        )

        print("\n============================================================")
        print("LIVE RTSP THERMAL ANALYSIS")
        print("============================================================")
        print(f"Resolution : {self.width} x {self.height}")
        print(f"Regions    : {len(self.regions)}")
        print(f"OCR rate   : ~{self.args.ocr_hz:.2f} Hz")
        print("\nActive:")
        print("  - 3x3 live hover")
        print("  - ROI MIN / AVG / MAX")
        print("  - Red hottest point per ROI")
        print("\nControls:")
        print("  SPACE  Freeze / return to latest live frame")
        print("  S      Screenshot")
        print("  R      Reconnect RTSP")
        print("  Q/ESC  Quit")

        try:
            while True:
                self._update_live_frame()

                display = self._render()

                cv2.imshow(
                    WINDOW_NAME,
                    display,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):
                    break

                if key == ord(" "):
                    self.frozen = (
                        not self.frozen
                    )

                    if self.frozen:
                        self.frozen_frame_id = (
                            self.current_frame_id
                        )

                    else:
                        # Immediately jump back to newest frame.
                        frame, frame_id, frame_time = (
                            self.capture.get_latest()
                        )

                        if frame is not None:
                            self.current_frame = frame
                            self.current_frame_id = frame_id
                            self.current_frame_time = frame_time
                            self.last_processed_id = frame_id
                            self._process_current_frame()

                elif key in (
                    ord("r"),
                    ord("R"),
                ):
                    self.capture.reconnect()

                elif key in (
                    ord("s"),
                    ord("S"),
                ):
                    self._save_screenshot(
                        display
                    )

                try:
                    if cv2.getWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_VISIBLE,
                    ) < 1:
                        break
                except cv2.error:
                    break

        finally:
            self.close()


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Live RTSP thermal analysis with first-time ROI/scale setup, "
            "continuous 3x3 hover, ROI MIN/AVG/MAX and red hotspots."
        )
    )

    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="RTSP URL. If omitted, the script prompts for it.",
    )

    parser.add_argument(
        "--preset",
        default=None,
        help=(
            "Reusable preset name. Default derives from RTSP path."
        ),
    )

    parser.add_argument(
        "--username",
        default=None,
        help="Optional RTSP username.",
    )

    parser.add_argument(
        "--password",
        default=None,
        help=(
            "Optional RTSP password. Prefer the secure prompt instead."
        ),
    )

    parser.add_argument(
        "--transport",
        choices=("tcp", "udp"),
        default="tcp",
    )

    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--tesseract",
        default=None,
        help="Optional full path to tesseract.exe",
    )

    parser.add_argument(
        "--ocr-hz",
        type=float,
        default=3.0,
        help="Live scale OCR frequency. Default: 3 Hz",
    )

    parser.add_argument(
        "--max-scale-jump",
        type=float,
        default=2.5,
        help=(
            "Normal accepted MIN/MAX change between OCR samples. "
            "Larger changes need two agreeing OCR samples. Default: 2.5 C"
        ),
    )

    parser.add_argument(
        "--jump-confirmation-tolerance",
        type=float,
        default=1.0,
        help=(
            "Two large-jump OCR samples must agree within this many C. "
            "Default: 1.0"
        ),
    )

    parser.add_argument(
        "--min-allowed-temp",
        type=float,
        default=-100.0,
    )

    parser.add_argument(
        "--max-allowed-temp",
        type=float,
        default=1000.0,
    )

    parser.add_argument(
        "--min-scale-span",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--decimal-places",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--ocr-padding",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--max-channel-difference",
        type=int,
        default=18,
    )

    parser.add_argument(
        "--erode",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--reconfigure",
        action="store_true",
        help="Ignore saved preset and annotate/configure again.",
    )

    parser.add_argument(
        "--preset-root",
        default="thermal-stream/presets",
        help="Where stream presets are saved.",
    )

    parser.add_argument(
        "--screenshot-dir",
        default="thermal-stream/screenshots",
    )

    return parser


def main():
    args = build_parser().parse_args()

    if args.ocr_hz <= 0:
        raise SystemExit("--ocr-hz must be > 0")

    if args.min_scale_span <= 0:
        raise SystemExit("--min-scale-span must be > 0")

    raw_url = args.url

    if not raw_url:
        raw_url = input("RTSP URL: ").strip()

    if not raw_url:
        raise SystemExit("No RTSP URL supplied.")

    parts = urlsplit(raw_url)
    url_has_credentials = "@" in parts.netloc

    username = args.username
    password = args.password

    if not url_has_credentials:
        if username is None:
            username = input(
                "Username (leave blank if none): "
            ).strip()

        if username and password is None:
            password = getpass.getpass(
                "Password: "
            )

    final_url = build_authenticated_rtsp_url(
        raw_url,
        username,
        password,
    )

    redacted_url = redact_rtsp_url(
        final_url
    )

    preset_name = safe_preset_name(
        args.preset
        or default_preset_name_from_url(
            final_url
        )
    )

    preset_root = (
        Path(args.preset_root)
        .expanduser()
        .resolve()
    )

    preset_dir = (
        preset_root
        / preset_name
    )

    configured_tesseract = configure_tesseract(
        args.tesseract
    )

    print("\nRTSP thermal configuration")
    print("--------------------------")
    print(f"Stream    : {redacted_url}")
    print(f"Transport : {args.transport.upper()}")
    print(f"Preset    : {preset_name}")
    print(f"Preset dir: {preset_dir}")

    if configured_tesseract:
        print(f"Tesseract : {configured_tesseract}")

    capture = RTSPCapture(
        final_url,
        transport=args.transport,
        reconnect_delay=args.reconnect_delay,
    )

    capture.start()

    try:
        first_frame = wait_for_first_frame(
            capture
        )

        regions_data, scale_config = (
            build_or_load_preset(
                first_frame,
                preset_dir,
                redacted_url,
                args,
            )
        )

        viewer = LiveThermalViewer(
            capture=capture,
            regions_data=regions_data,
            scale_config=scale_config,
            preset_dir=preset_dir,
            args=args,
        )

        viewer.run()

    finally:
        capture.stop()


if __name__ == "__main__":
    main()
