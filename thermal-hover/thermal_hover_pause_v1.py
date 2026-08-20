"""
Interactive thermal-video player with mouse-hover temperature inspection.

Main use case
-------------
Run this script with a thermal MP4. It opens a small video player.

- While the video is PLAYING:
    temperature hover is disabled.

- While the video is PAUSED:
    move the mouse over the thermal image and the program shows the
    estimated temperature at that pixel.

The temperature calculation deliberately reuses the same thermal-scale
logic as the existing Engro Step 02 / Step 03 pipeline.

Preferred scale source
----------------------
1. If Step 03 already produced scale_readings.csv for the video, use the
   cleaned MIN/MAX values from that exact frame. This is fast and keeps
   the viewer consistent with Step 03.

2. If no Step 03 reading exists for the paused frame, OCR the current
   frame on demand using the Step 02 scale configuration.

Important limitation
--------------------
The input MP4 is a rendered thermal video, not raw radiometric sensor data.
Therefore the displayed value is an ESTIMATED temperature reconstructed
from the on-screen thermal scale, exactly like Step 03.

Colored UI overlays (green labels, red markers, etc.) are not thermal scene
pixels, so the viewer reports them as N/A rather than inventing a value.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
import sys
from pathlib import Path

# Allow this archived V1 file to live inside thermal-hover/ while still
# importing the parent repo's Step-02/Step-03 modules.
REPO_PARENT = Path(__file__).resolve().parent.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))

import cv2
import numpy as np
import pytesseract

# Reuse the exact OCR configuration logic already developed for Step 02.
from step_02_configure_thermal_scale import (
    OCR_METHODS_TO_TEST,
    choose_best_pair,
    run_one_ocr,
)

# Reuse the exact grayscale-bar -> temperature mapping already used by Step 03.
from step_03_process_thermal_temperatures import (
    build_grayscale_temperature_lut,
)

# Shared JSON helper from the provenance-aware pipeline.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import step_code as ps


WINDOW_NAME = "Engro Thermal Mouse Hover Viewer"

DEFAULT_STEP_02_ROOTS = (
    "step-02-ocr-scale-config",
    "step02-ocr-scale-config",
    "step-2-ocr-scale-config",
)

DEFAULT_STEP_03_ROOTS = (
    "step-03-roi-videos",
    "step03-roi-videos",
    "step-3-roi-videos",
)

# Display limits only affect the GUI.
# All temperature lookups remain in ORIGINAL video coordinates.
MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 850

# OpenCV BGR colours.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)


# =============================================================================
# PATH DISCOVERY
# =============================================================================

def find_named_ancestor(path: Path, folder_name: str):
    """Return the nearest ancestor whose folder name matches folder_name."""
    path = Path(path).resolve()

    for parent in path.parents:
        if parent.name.lower() == folder_name.lower():
            return parent

    return None


def infer_project_root(video_path: Path) -> Path:
    """
    Try to infer the thermal-project root.

    Normal layout:
        thermal-plant/
            raw-videos/
                ...
            step-02-ocr-scale-config/
            step-03-roi-videos/

    If the video is not under raw-videos, fall back to the current directory.
    """
    raw_root = find_named_ancestor(video_path, "raw-videos")

    if raw_root is not None:
        return raw_root.parent

    return Path.cwd().resolve()


def relative_video_parent(video_path: Path):
    """
    Return the relative parent used by the Step 02 / Step 03 folder structure.

    Example:
        raw-videos/Furnace/Camera-01/video.mp4
    gives:
        Furnace/Camera-01
    """
    raw_root = find_named_ancestor(video_path, "raw-videos")

    if raw_root is None:
        return Path()

    relative_video = video_path.resolve().relative_to(raw_root)
    return relative_video.parent


def expected_stage_file(
    project_root: Path,
    video_path: Path,
    root_names,
    filename: str,
):
    """
    Try the standard output-root naming variants used during this project.

    Returns the first existing file or None.
    """
    relative_parent = relative_video_parent(video_path)

    for root_name in root_names:
        candidate = (
            project_root
            / root_name
            / relative_parent
            / video_path.stem
            / filename
        )

        if candidate.is_file():
            return candidate.resolve()

    return None


def unique_fallback_search(project_root: Path, video_stem: str, filename: str):
    """
    Last-resort lookup for older folder naming.

    We only accept the result when exactly one matching file exists beneath
    a folder named after the video stem. This avoids silently choosing the
    wrong configuration if multiple copies exist.
    """
    matches = []

    for path in project_root.rglob(filename):
        if path.parent.name == video_stem:
            matches.append(path.resolve())

    if len(matches) == 1:
        return matches[0]

    return None


def resolve_scale_config(video_path: Path, project_root: Path, override):
    """Find Step 02 scale_config.json."""
    if override:
        path = Path(override).resolve()

        if not path.is_file():
            raise FileNotFoundError(f"Scale config does not exist: {path}")

        return path

    path = expected_stage_file(
        project_root,
        video_path,
        DEFAULT_STEP_02_ROOTS,
        "scale_config.json",
    )

    if path is not None:
        return path

    return unique_fallback_search(
        project_root,
        video_path.stem,
        "scale_config.json",
    )


def resolve_scale_readings(video_path: Path, project_root: Path, override):
    """Find Step 03 scale_readings.csv when available."""
    if override:
        path = Path(override).resolve()

        if not path.is_file():
            raise FileNotFoundError(f"Scale readings do not exist: {path}")

        return path

    path = expected_stage_file(
        project_root,
        video_path,
        DEFAULT_STEP_03_ROOTS,
        "scale_readings.csv",
    )

    if path is not None:
        return path

    return unique_fallback_search(
        project_root,
        video_path.stem,
        "scale_readings.csv",
    )


# =============================================================================
# STEP 03 SCALE-READING CACHE
# =============================================================================

def load_scale_readings_csv(path: Path, total_frames: int):
    """
    Load Step 03's cleaned MIN/MAX scale values.

    We use NumPy arrays instead of a large dictionary so even long videos
    remain memory-efficient.
    """
    clean_min = np.full(
        total_frames,
        np.nan,
        dtype=np.float64,
    )

    clean_max = np.full(
        total_frames,
        np.nan,
        dtype=np.float64,
    )

    loaded_rows = 0

    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required = {
            "frame",
            "clean_min_temp_used",
            "clean_max_temp_used",
        }

        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(
                "scale_readings.csv does not contain the expected Step-03 "
                "columns: frame, clean_min_temp_used, clean_max_temp_used"
            )

        for row in reader:
            try:
                frame_number = int(row["frame"])

                if not (0 <= frame_number < total_frames):
                    continue

                min_temp = float(row["clean_min_temp_used"])
                max_temp = float(row["clean_max_temp_used"])
            except (TypeError, ValueError):
                continue

            if not (
                np.isfinite(min_temp)
                and np.isfinite(max_temp)
                and max_temp > min_temp
            ):
                continue

            clean_min[frame_number] = min_temp
            clean_max[frame_number] = max_temp
            loaded_rows += 1

    return clean_min, clean_max, loaded_rows


# =============================================================================
# OCR FALLBACK
# =============================================================================

def configure_tesseract(explicit_path):
    """
    Configure Tesseract when a path is supplied.

    If no path is supplied, try the common Windows installation path.
    If neither works, OCR remains unavailable, but Step-03 cached readings
    can still make the viewer fully usable.
    """
    if explicit_path:
        path = Path(explicit_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Tesseract executable does not exist: {path}"
            )

        pytesseract.pytesseract.tesseract_cmd = str(path)
        return str(path)

    common_windows = Path(
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    if os.name == "nt" and common_windows.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(common_windows)
        return str(common_windows)

    return None


def tesseract_is_available():
    """Check Tesseract only when live OCR is actually needed."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_scale_for_frame(
    frame,
    scale_config,
    min_allowed,
    max_allowed,
    min_scale_span,
):
    """
    OCR the paused frame on demand.

    Unlike Step 03's temporal scan, the user may jump directly to any frame.
    Therefore this fallback uses Step 02's multi-method plausibility chooser
    without assuming that the current scale must be close to frame zero.
    """
    scale = scale_config["scale"]

    min_roi = scale["min_text_roi"]
    max_roi = scale["max_text_roi"]

    decimal_places = int(
        scale.get("display_decimal_places", 1)
    )

    padding = int(
        scale.get("ocr_padding_pixels", 6)
    )

    # Start with the methods selected during Step 02, because those are the
    # most likely to work for this specific camera/video.
    preferred = [
        (
            scale.get("min_ocr_method", "green_difference"),
            int(scale.get("min_ocr_psm", 7)),
        ),
        (
            scale.get("max_ocr_method", "green_difference"),
            int(scale.get("max_ocr_psm", 7)),
        ),
    ]

    # Then add the full Step-02 method set, skipping duplicates.
    methods = []
    seen = set()

    for method, psm in [*preferred, *OCR_METHODS_TO_TEST]:
        key = (method, int(psm))

        if key in seen:
            continue

        seen.add(key)
        methods.append(key)

    min_candidates = []
    max_candidates = []

    for method, psm in methods:
        min_candidates.append(
            run_one_ocr(
                frame,
                min_roi,
                method=method,
                psm=psm,
                decimal_places=decimal_places,
                padding=padding,
            )
        )

        max_candidates.append(
            run_one_ocr(
                frame,
                max_roi,
                method=method,
                psm=psm,
                decimal_places=decimal_places,
                padding=padding,
            )
        )

    best_pair = choose_best_pair(
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        min_allowed=float(min_allowed),
        max_allowed=float(max_allowed),
        min_scale_span=float(min_scale_span),
    )

    if best_pair is None:
        return None

    best_min, best_max = best_pair

    return {
        "min_temp": float(best_min["value"]),
        "max_temp": float(best_max["value"]),
        "source": "LIVE OCR",
        "min_text": best_min["raw_text"],
        "max_text": best_max["raw_text"],
        "min_method": f"{best_min['method']}/PSM{best_min['psm']}",
        "max_method": f"{best_max['method']}/PSM{best_max['psm']}",
    }


# =============================================================================
# TEMPERATURE MAP
# =============================================================================

def build_temperature_map(
    frame,
    scale_config,
    min_temp,
    max_temp,
):
    """
    Build one full-resolution estimated temperature map for the PAUSED frame.

    This operation is cheap once MIN/MAX are known:
      current bar -> 256-entry LUT -> array lookup for every pixel.
    """
    scale = scale_config["scale"]

    bar_roi = scale["bar_roi"]

    inner_fraction = float(
        scale.get(
            "bar_horizontal_inner_fraction",
            0.20,
        )
    )

    lut = build_grayscale_temperature_lut(
        frame,
        bar_roi=bar_roi,
        min_temp=float(min_temp),
        max_temp=float(max_temp),
        inner_fraction=inner_fraction,
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    return lut[gray]


def pixel_is_neutral_thermal(frame, x, y, max_channel_difference):
    """
    Return True when a pixel is approximately grayscale.

    The rendered thermal scene is grayscale in the current footage.
    Strongly coloured pixels are usually UI overlays rather than thermal data.
    """
    b, g, r = [
        int(value)
        for value in frame[y, x]
    ]

    return (
        max(b, g, r) - min(b, g, r)
        <= int(max_channel_difference)
    )


def sample_temperature(
    temperature_map,
    frame,
    x,
    y,
    hover_radius,
    max_channel_difference,
):
    """
    Read the temperature under the cursor.

    hover_radius = 0:
        exact requested pixel.

    hover_radius > 0:
        optional small neighbourhood mean, useful later if compression noise
        makes one-pixel readings jump around.
    """
    height, width = temperature_map.shape[:2]

    if not (0 <= x < width and 0 <= y < height):
        return None

    if hover_radius <= 0:
        if not pixel_is_neutral_thermal(
            frame,
            x,
            y,
            max_channel_difference,
        ):
            return {
                "valid": False,
                "reason": "COLORED UI / OVERLAY PIXEL",
            }

        return {
            "valid": True,
            "temperature": float(
                temperature_map[y, x]
            ),
            "sample_count": 1,
        }

    radius = int(hover_radius)

    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(width, x + radius + 1)
    y2 = min(height, y + radius + 1)

    patch_temp = temperature_map[
        y1:y2,
        x1:x2,
    ]

    patch_frame = frame[
        y1:y2,
        x1:x2,
    ].astype(np.int16)

    channel_spread = (
        patch_frame.max(axis=2)
        - patch_frame.min(axis=2)
    )

    valid = (
        channel_spread
        <= int(max_channel_difference)
    )

    values = patch_temp[valid]

    if values.size == 0:
        return {
            "valid": False,
            "reason": "NO THERMAL PIXELS IN NEIGHBOURHOOD",
        }

    return {
        "valid": True,
        "temperature": float(np.mean(values)),
        "sample_count": int(values.size),
    }


# =============================================================================
# DRAWING HELPERS
# =============================================================================

def draw_text_box(
    image,
    lines,
    origin=(12, 12),
    text_color=WHITE,
    background=BLACK,
    font_scale=0.56,
):
    """Draw a compact readable multi-line box."""
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    measurements = [
        cv2.getTextSize(
            str(line),
            font,
            font_scale,
            thickness,
        )
        for line in lines
    ]

    widths = [
        size[0][0]
        for size in measurements
    ]

    heights = [
        size[0][1] + size[1]
        for size in measurements
    ]

    line_height = max(heights, default=18) + 7

    x, y = origin

    box_width = max(widths, default=0) + 18
    box_height = line_height * len(lines) + 12

    cv2.rectangle(
        image,
        (x, y),
        (
            min(image.shape[1] - 1, x + box_width),
            min(image.shape[0] - 1, y + box_height),
        ),
        background,
        -1,
    )

    text_y = y + line_height

    for line in lines:
        cv2.putText(
            image,
            str(line),
            (x + 9, text_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

        text_y += line_height


def draw_hover_tooltip(
    image,
    display_x,
    display_y,
    lines,
    valid=True,
):
    """Draw cursor crosshair + temperature tooltip."""
    color = GREEN if valid else RED

    cv2.drawMarker(
        image,
        (display_x, display_y),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=22,
        thickness=2,
    )

    # Estimate a compact tooltip size.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_height = 21

    widths = []

    for line in lines:
        (w, _), _ = cv2.getTextSize(
            line,
            font,
            font_scale,
            thickness,
        )
        widths.append(w)

    tooltip_width = max(widths, default=100) + 18
    tooltip_height = line_height * len(lines) + 10

    # Prefer drawing down/right of the cursor.
    x1 = display_x + 15
    y1 = display_y + 15

    # Flip to the other side if we would leave the window.
    if x1 + tooltip_width >= image.shape[1]:
        x1 = max(0, display_x - tooltip_width - 15)

    if y1 + tooltip_height >= image.shape[0]:
        y1 = max(0, display_y - tooltip_height - 15)

    x2 = min(
        image.shape[1] - 1,
        x1 + tooltip_width,
    )

    y2 = min(
        image.shape[0] - 1,
        y1 + tooltip_height,
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

    y = y1 + 19

    for line in lines:
        cv2.putText(
            image,
            line,
            (x1 + 8, y),
            font,
            font_scale,
            WHITE,
            thickness,
            cv2.LINE_AA,
        )

        y += line_height


# =============================================================================
# PLAYER
# =============================================================================

class ThermalHoverPlayer:
    def __init__(
        self,
        video_path,
        scale_config_path,
        scale_readings_path,
        args,
    ):
        self.video_path = Path(video_path).resolve()
        self.scale_config_path = Path(
            scale_config_path
        ).resolve()

        self.scale_readings_path = (
            Path(scale_readings_path).resolve()
            if scale_readings_path is not None
            else None
        )

        self.args = args

        self.scale_config = ps.load_json(
            self.scale_config_path
        )

        if not isinstance(
            self.scale_config,
            dict,
        ):
            raise RuntimeError(
                f"Could not read scale config: {self.scale_config_path}"
            )

        # ---------------------------------------------------------------------
        # Open video and read basic metadata.
        # ---------------------------------------------------------------------
        self.cap = cv2.VideoCapture(
            str(self.video_path)
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {self.video_path}"
            )

        self.fps = float(
            self.cap.get(cv2.CAP_PROP_FPS)
        )

        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        self.total_frames = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if self.fps <= 0 or self.total_frames <= 0:
            self.cap.release()
            raise RuntimeError(
                "Video reported invalid FPS/frame count."
            )

        # ---------------------------------------------------------------------
        # Validate that Step 02 was configured for this resolution.
        # ---------------------------------------------------------------------
        cfg_width = int(
            self.scale_config["video"]["width"]
        )

        cfg_height = int(
            self.scale_config["video"]["height"]
        )

        if (
            self.width,
            self.height,
        ) != (
            cfg_width,
            cfg_height,
        ):
            self.cap.release()

            raise RuntimeError(
                "Video resolution does not match scale_config.json.\n"
                f"Video       : {self.width} x {self.height}\n"
                f"Scale config: {cfg_width} x {cfg_height}"
            )

        # ---------------------------------------------------------------------
        # Optional Step-03 cached scale readings.
        # ---------------------------------------------------------------------
        self.scale_min_cache = np.full(
            self.total_frames,
            np.nan,
            dtype=np.float64,
        )

        self.scale_max_cache = np.full(
            self.total_frames,
            np.nan,
            dtype=np.float64,
        )

        self.cached_scale_rows = 0

        if self.scale_readings_path is not None:
            (
                self.scale_min_cache,
                self.scale_max_cache,
                self.cached_scale_rows,
            ) = load_scale_readings_csv(
                self.scale_readings_path,
                self.total_frames,
            )

        # Live OCR results are cached after the first pause on a frame.
        self.live_ocr_cache = {}

        # ---------------------------------------------------------------------
        # Player state.
        # ---------------------------------------------------------------------
        self.frame_index = 0
        self.current_frame = None

        self.playing = False

        # Temperature map is only built for a paused frame.
        self.temperature_map = None
        self.temperature_map_frame = None
        self.current_scale_info = None

        # Mouse coordinates:
        # display-space values come from OpenCV callback;
        # original-space values are used for temperature lookup.
        self.mouse_display_x = None
        self.mouse_display_y = None
        self.mouse_original_x = None
        self.mouse_original_y = None

        self.display_width = None
        self.display_height = None

        # Trackbar callbacks fire when we update them programmatically.
        # This flag prevents our own update from being interpreted as
        # a user seek.
        self.updating_trackbar = False
        self.pending_seek = None

        # Used for approximately real-time playback pacing.
        self.next_playback_deadline = None

        self._load_frame(0)

    # -------------------------------------------------------------------------
    # VIDEO SEEK / LOAD
    # -------------------------------------------------------------------------

    def _load_frame(self, frame_number):
        """Random-access load of one frame."""
        frame_number = int(
            np.clip(
                frame_number,
                0,
                self.total_frames - 1,
            )
        )

        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number,
        )

        ok, frame = self.cap.read()

        if not ok:
            raise RuntimeError(
                f"Could not read frame {frame_number}"
            )

        self.frame_index = frame_number
        self.current_frame = frame

        self.temperature_map = None
        self.temperature_map_frame = None
        self.current_scale_info = None

        self._clear_hover()

    def _read_next_frame(self):
        """Sequential read used during playback."""
        if self.frame_index >= self.total_frames - 1:
            self.playing = False
            return False

        ok, frame = self.cap.read()

        if not ok:
            self.playing = False
            return False

        self.frame_index += 1
        self.current_frame = frame

        self.temperature_map = None
        self.temperature_map_frame = None
        self.current_scale_info = None

        self._clear_hover()

        return True

    def _seek(self, frame_number):
        """Seek and remain paused."""
        self.playing = False
        self.next_playback_deadline = None

        self._load_frame(frame_number)

        # Once the user deliberately seeks to a frame, immediately prepare
        # the temperature map because the player is paused.
        self._ensure_temperature_map()

    # -------------------------------------------------------------------------
    # SCALE / TEMPERATURE
    # -------------------------------------------------------------------------

    def _get_scale_for_current_frame(self):
        """
        Get trusted MIN/MAX for the current frame.

        Priority:
          1. Step 03 clean scale_readings.csv
          2. previously cached live OCR
          3. new live OCR attempt
        """
        idx = self.frame_index

        cached_min = self.scale_min_cache[idx]
        cached_max = self.scale_max_cache[idx]

        if (
            np.isfinite(cached_min)
            and np.isfinite(cached_max)
            and cached_max > cached_min
        ):
            return {
                "min_temp": float(cached_min),
                "max_temp": float(cached_max),
                "source": "STEP 03 CLEAN SCALE",
            }

        if idx in self.live_ocr_cache:
            return self.live_ocr_cache[idx]

        # No Step-03 value for this frame.
        # Try OCR only when Tesseract is available.
        if not tesseract_is_available():
            return {
                "error": (
                    "No Step-03 scale reading for this frame and "
                    "Tesseract is unavailable for live OCR."
                )
            }

        print(
            f"\nReading thermal scale for paused frame "
            f"{idx}/{self.total_frames - 1}..."
        )

        scale_info = ocr_scale_for_frame(
            self.current_frame,
            self.scale_config,
            min_allowed=self.args.min_allowed_temp,
            max_allowed=self.args.max_allowed_temp,
            min_scale_span=self.args.min_scale_span,
        )

        if scale_info is None:
            scale_info = {
                "error": (
                    "Could not obtain a plausible MIN/MAX scale "
                    "from this paused frame."
                )
            }

        self.live_ocr_cache[idx] = scale_info

        return scale_info

    def _ensure_temperature_map(self):
        """
        Build the temperature map once for the current paused frame.

        Mouse movement itself will not re-run OCR or rebuild the map.
        """
        if self.playing:
            return

        if (
            self.temperature_map is not None
            and self.temperature_map_frame == self.frame_index
        ):
            return

        scale_info = self._get_scale_for_current_frame()
        self.current_scale_info = scale_info

        if "error" in scale_info:
            self.temperature_map = None
            self.temperature_map_frame = self.frame_index
            return

        try:
            self.temperature_map = build_temperature_map(
                self.current_frame,
                self.scale_config,
                min_temp=scale_info["min_temp"],
                max_temp=scale_info["max_temp"],
            )

            self.temperature_map_frame = (
                self.frame_index
            )

        except Exception as exc:
            self.temperature_map = None
            self.temperature_map_frame = (
                self.frame_index
            )

            self.current_scale_info = {
                "error": (
                    "Could not build temperature map: "
                    f"{exc}"
                )
            }

    # -------------------------------------------------------------------------
    # MOUSE
    # -------------------------------------------------------------------------

    def _clear_hover(self):
        self.mouse_display_x = None
        self.mouse_display_y = None
        self.mouse_original_x = None
        self.mouse_original_y = None

    def _mouse_callback(self, event, x, y, flags, param):
        """
        Mouse-hover handler.

        While playing:
            intentionally ignore hover.

        While paused:
            map display coordinates back to ORIGINAL video pixels.
        """
        if self.playing:
            self._clear_hover()
            return

        if event != cv2.EVENT_MOUSEMOVE:
            return

        if (
            self.display_width is None
            or self.display_height is None
        ):
            return

        if not (
            0 <= x < self.display_width
            and 0 <= y < self.display_height
        ):
            self._clear_hover()
            return

        original_x = int(
            round(
                x
                * self.width
                / self.display_width
            )
        )

        original_y = int(
            round(
                y
                * self.height
                / self.display_height
            )
        )

        original_x = max(
            0,
            min(
                original_x,
                self.width - 1,
            ),
        )

        original_y = max(
            0,
            min(
                original_y,
                self.height - 1,
            ),
        )

        self.mouse_display_x = int(x)
        self.mouse_display_y = int(y)
        self.mouse_original_x = original_x
        self.mouse_original_y = original_y

    # -------------------------------------------------------------------------
    # DISPLAY
    # -------------------------------------------------------------------------

    def _calculate_display_size(self):
        """Fit the video on screen without enlarging it."""
        scale = min(
            MAX_DISPLAY_WIDTH / self.width,
            MAX_DISPLAY_HEIGHT / self.height,
            1.0,
        )

        display_width = max(
            1,
            int(round(self.width * scale)),
        )

        display_height = max(
            1,
            int(round(self.height * scale)),
        )

        return display_width, display_height

    def _render_display(self):
        """Build the current GUI image."""
        self.display_width, self.display_height = (
            self._calculate_display_size()
        )

        display = cv2.resize(
            self.current_frame,
            (
                self.display_width,
                self.display_height,
            ),
            interpolation=(
                cv2.INTER_AREA
                if self.display_width < self.width
                else cv2.INTER_LINEAR
            ),
        )

        timestamp = (
            self.frame_index / self.fps
        )

        duration = (
            self.total_frames / self.fps
        )

        state = (
            "PLAYING"
            if self.playing
            else "PAUSED - HOVER ACTIVE"
        )

        state_colour = (
            YELLOW
            if self.playing
            else GREEN
        )

        # ---------------------------------------------------------------------
        # Main status panel.
        # ---------------------------------------------------------------------
        status_lines = [
            f"{state}",
            (
                f"Frame {self.frame_index:,}/{self.total_frames - 1:,} | "
                f"{timestamp:.2f}s / {duration:.2f}s"
            ),
        ]

        if not self.playing:
            if (
                self.current_scale_info is not None
                and "error" not in self.current_scale_info
            ):
                status_lines.append(
                    (
                        f"Scale: {self.current_scale_info['min_temp']:.1f} C "
                        f"to {self.current_scale_info['max_temp']:.1f} C | "
                        f"{self.current_scale_info['source']}"
                    )
                )

            elif (
                self.current_scale_info is not None
                and "error" in self.current_scale_info
            ):
                status_lines.append(
                    "Temperature unavailable: "
                    + self.current_scale_info["error"]
                )

        draw_text_box(
            display,
            status_lines,
            origin=(12, 12),
            text_color=state_colour,
        )

        # ---------------------------------------------------------------------
        # Hover readout.
        # ---------------------------------------------------------------------
        if (
            not self.playing
            and self.mouse_original_x is not None
            and self.mouse_original_y is not None
        ):
            x = self.mouse_original_x
            y = self.mouse_original_y

            if self.temperature_map is None:
                tooltip_lines = [
                    f"x={x}, y={y}",
                    "Temperature unavailable",
                ]

                draw_hover_tooltip(
                    display,
                    self.mouse_display_x,
                    self.mouse_display_y,
                    tooltip_lines,
                    valid=False,
                )

            else:
                sample = sample_temperature(
                    self.temperature_map,
                    self.current_frame,
                    x,
                    y,
                    hover_radius=self.args.hover_radius,
                    max_channel_difference=(
                        self.args.max_channel_difference
                    ),
                )

                if sample is None:
                    return display

                if sample["valid"]:
                    gray_value = int(
                        cv2.cvtColor(
                            self.current_frame[
                                y:y + 1,
                                x:x + 1,
                            ],
                            cv2.COLOR_BGR2GRAY,
                        )[0, 0]
                    )

                    if self.args.hover_radius <= 0:
                        temperature_label = (
                            f"Estimated: "
                            f"{sample['temperature']:.1f} C"
                        )
                    else:
                        side = (
                            2
                            * int(self.args.hover_radius)
                            + 1
                        )

                        temperature_label = (
                            f"Estimated {side}x{side} avg: "
                            f"{sample['temperature']:.1f} C"
                        )

                    tooltip_lines = [
                        temperature_label,
                        f"Pixel: x={x}, y={y}",
                        f"Gray intensity: {gray_value}",
                    ]

                    draw_hover_tooltip(
                        display,
                        self.mouse_display_x,
                        self.mouse_display_y,
                        tooltip_lines,
                        valid=True,
                    )

                else:
                    tooltip_lines = [
                        f"Pixel: x={x}, y={y}",
                        sample["reason"],
                        "Temperature: N/A",
                    ]

                    draw_hover_tooltip(
                        display,
                        self.mouse_display_x,
                        self.mouse_display_y,
                        tooltip_lines,
                        valid=False,
                    )

        # ---------------------------------------------------------------------
        # Always-visible control hint at bottom.
        # ---------------------------------------------------------------------
        controls = (
            "SPACE Play/Pause | A/D +/-1 frame | "
            "J/L +/-1 sec | R Start | Q/ESC Quit | "
            "Timeline slider = seek"
        )

        (text_w, text_h), baseline = cv2.getTextSize(
            controls,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            1,
        )

        y1 = max(
            0,
            display.shape[0] - text_h - 16,
        )

        cv2.rectangle(
            display,
            (0, y1),
            (
                min(
                    display.shape[1] - 1,
                    text_w + 20,
                ),
                display.shape[0] - 1,
            ),
            BLACK,
            -1,
        )

        cv2.putText(
            display,
            controls,
            (
                8,
                display.shape[0] - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            WHITE,
            1,
            cv2.LINE_AA,
        )

        return display

    # -------------------------------------------------------------------------
    # TRACKBAR
    # -------------------------------------------------------------------------

    def _trackbar_callback(self, position):
        if self.updating_trackbar:
            return

        self.pending_seek = int(position)

    def _update_trackbar(self):
        self.updating_trackbar = True

        cv2.setTrackbarPos(
            "Frame",
            WINDOW_NAME,
            int(self.frame_index),
        )

        self.updating_trackbar = False

    # -------------------------------------------------------------------------
    # PLAYBACK
    # -------------------------------------------------------------------------

    def _toggle_play_pause(self):
        if self.playing:
            # Pause on the currently visible frame.
            self.playing = False
            self.next_playback_deadline = None

            self._ensure_temperature_map()

        else:
            # Resume sequential decoding from the NEXT frame.
            if self.frame_index >= self.total_frames - 1:
                self._load_frame(0)

            self.cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                self.frame_index + 1,
            )

            self.playing = True
            self.next_playback_deadline = (
                time.perf_counter()
            )

            self._clear_hover()

    def _advance_playback_if_due(self):
        """
        Decode frames close to the video's natural FPS.

        The player is intentionally simple, but this prevents a 25-FPS video
        from blasting through as fast as the CPU can decode.
        """
        if not self.playing:
            return

        now = time.perf_counter()

        if self.next_playback_deadline is None:
            self.next_playback_deadline = now

        if now < self.next_playback_deadline:
            return

        if not self._read_next_frame():
            self._ensure_temperature_map()
            return

        frame_period = (
            1.0 / self.fps
        )

        self.next_playback_deadline += (
            frame_period
        )

        # If decoding/display falls far behind, do not build an ever-growing
        # timing debt. Resume from "now".
        if (
            now - self.next_playback_deadline
            > 0.5
        ):
            self.next_playback_deadline = (
                now + frame_period
            )

    # -------------------------------------------------------------------------
    # MAIN GUI LOOP
    # -------------------------------------------------------------------------

    def run(self):
        """Open the interactive player."""
        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_AUTOSIZE,
        )

        cv2.createTrackbar(
            "Frame",
            WINDOW_NAME,
            0,
            max(
                1,
                self.total_frames - 1,
            ),
            self._trackbar_callback,
        )

        cv2.setMouseCallback(
            WINDOW_NAME,
            self._mouse_callback,
        )

        # Start paused so the first frame is immediately inspectable.
        self._ensure_temperature_map()

        print("\n============================================")
        print("THERMAL MOUSE-HOVER VIEWER")
        print("============================================")
        print(f"Video        : {self.video_path}")
        print(f"Resolution   : {self.width} x {self.height}")
        print(f"FPS          : {self.fps:.3f}")
        print(f"Frames       : {self.total_frames}")
        print(f"Scale config : {self.scale_config_path}")

        if self.scale_readings_path is not None:
            print(
                f"Scale CSV    : {self.scale_readings_path}"
            )

            print(
                f"Cached frames: {self.cached_scale_rows}"
            )
        else:
            print(
                "Scale CSV    : not found; paused frames will use live OCR"
            )

        print("\nControls:")
        print("  SPACE -> Play / pause")
        print("  A     -> Previous frame")
        print("  D     -> Next frame")
        print("  J     -> Back 1 second")
        print("  L     -> Forward 1 second")
        print("  R     -> Return to frame 0")
        print("  Slider-> Seek to any frame")
        print("  Q/ESC -> Close")
        print(
            "\nMouse temperature is active only while PAUSED."
        )

        try:
            while True:
                # -------------------------------------------------------------
                # Process a seek requested through the frame slider.
                # -------------------------------------------------------------
                if self.pending_seek is not None:
                    requested = self.pending_seek
                    self.pending_seek = None

                    if requested != self.frame_index:
                        self._seek(requested)

                # -------------------------------------------------------------
                # Advance video if playing and enough time has elapsed.
                # -------------------------------------------------------------
                self._advance_playback_if_due()

                # -------------------------------------------------------------
                # Display current frame.
                # -------------------------------------------------------------
                display = self._render_display()

                cv2.imshow(
                    WINDOW_NAME,
                    display,
                )

                self._update_trackbar()

                # waitKeyEx gives us better keyboard compatibility than waitKey.
                key = cv2.waitKeyEx(10)

                if key == -1:
                    continue

                # ESC or Q.
                if key in (
                    27,
                    ord("q"),
                    ord("Q"),
                ):
                    break

                # SPACE.
                if key == 32:
                    self._toggle_play_pause()
                    continue

                # A = previous frame.
                if key in (
                    ord("a"),
                    ord("A"),
                ):
                    self._seek(
                        self.frame_index - 1
                    )
                    continue

                # D = next frame.
                if key in (
                    ord("d"),
                    ord("D"),
                ):
                    self._seek(
                        self.frame_index + 1
                    )
                    continue

                # J = back approximately one second.
                if key in (
                    ord("j"),
                    ord("J"),
                ):
                    self._seek(
                        self.frame_index
                        - int(round(self.fps))
                    )
                    continue

                # L = forward approximately one second.
                if key in (
                    ord("l"),
                    ord("L"),
                ):
                    self._seek(
                        self.frame_index
                        + int(round(self.fps))
                    )
                    continue

                # R = reset.
                if key in (
                    ord("r"),
                    ord("R"),
                ):
                    self._seek(0)
                    continue

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Open a thermal video in an interactive player. "
            "When paused, hover the mouse over thermal-scene pixels "
            "to see their estimated temperature."
        )
    )

    # This is intentionally the only REQUIRED parameter.
    parser.add_argument(
        "video",
        help="Path to the thermal MP4 to inspect",
    )

    parser.add_argument(
        "--scale-config",
        default=None,
        help=(
            "Optional explicit Step-02 scale_config.json. "
            "Normally auto-discovered from the video path."
        ),
    )

    parser.add_argument(
        "--scale-readings",
        default=None,
        help=(
            "Optional explicit Step-03 scale_readings.csv. "
            "Normally auto-discovered when it exists."
        ),
    )

    parser.add_argument(
        "--tesseract",
        default=None,
        help=(
            "Optional Tesseract executable path. Needed only when a paused "
            "frame has no Step-03 cached scale reading and live OCR is required."
        ),
    )

    parser.add_argument(
        "--hover-radius",
        type=int,
        default=0,
        help=(
            "0 = exact single pixel (default). "
            "1 = average a 3x3 neighbourhood, 2 = 5x5, etc."
        ),
    )

    parser.add_argument(
        "--max-channel-difference",
        type=int,
        default=18,
        help=(
            "Pixels whose B/G/R spread exceeds this are treated as colored "
            "camera UI rather than thermal-scene pixels. Default: 18."
        ),
    )

    parser.add_argument(
        "--min-allowed-temp",
        type=float,
        default=-100.0,
        help="Live-OCR plausibility lower bound. Default: -100 C.",
    )

    parser.add_argument(
        "--max-allowed-temp",
        type=float,
        default=1000.0,
        help="Live-OCR plausibility upper bound. Default: 1000 C.",
    )

    parser.add_argument(
        "--min-scale-span",
        type=float,
        default=None,
        help=(
            "Minimum plausible MAX-MIN span for live OCR. "
            "Default uses Step-02 config, otherwise 5 C."
        ),
    )

    args = parser.parse_args()

    args.hover_radius = max(
        0,
        int(args.hover_radius),
    )

    video_path = Path(
        args.video
    ).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video does not exist: {video_path}"
        )

    project_root = infer_project_root(
        video_path
    )

    scale_config_path = resolve_scale_config(
        video_path,
        project_root,
        args.scale_config,
    )

    if scale_config_path is None:
        raise RuntimeError(
            "Could not automatically find Step-02 scale_config.json.\n"
            "Run Step 02 for this video first, or pass:\n"
            "  --scale-config \"path\\to\\scale_config.json\""
        )

    scale_readings_path = resolve_scale_readings(
        video_path,
        project_root,
        args.scale_readings,
    )

    # Configure Tesseract if the user supplied a path or if the normal
    # Windows install exists. It will only actually be called if needed.
    configured_tesseract = configure_tesseract(
        args.tesseract
    )

    scale_config = ps.load_json(
        scale_config_path
    )

    if args.min_scale_span is None:
        args.min_scale_span = float(
            scale_config
            .get("scale", {})
            .get("min_scale_span_c", 5.0)
        )

    print(
        f"\nProject root inferred as:\n{project_root}"
    )

    print(
        f"\nUsing Step-02 configuration:\n{scale_config_path}"
    )

    if scale_readings_path is not None:
        print(
            f"\nUsing Step-03 cleaned scale readings when available:\n"
            f"{scale_readings_path}"
        )
    else:
        print(
            "\nNo Step-03 scale_readings.csv found. "
            "Paused frames will use live OCR."
        )

    if configured_tesseract:
        print(
            f"\nTesseract configured:\n{configured_tesseract}"
        )

    player = ThermalHoverPlayer(
        video_path=video_path,
        scale_config_path=scale_config_path,
        scale_readings_path=scale_readings_path,
        args=args,
    )

    player.run()


if __name__ == "__main__":
    main()
