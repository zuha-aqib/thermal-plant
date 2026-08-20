"""
Engro Thermal Hover - continuous recorded-video viewer
======================================================

Purpose
-------
Play a rendered thermal MP4 continuously (auto-looping by default) and show
an estimated 3x3-neighbourhood temperature under the mouse while the video is
PLAYING or PAUSED.

The script is intentionally SELF-CONTAINED. It does not import the Engro
Step-02/Step-03 scripts or pipeline_state.py.

It supports two modes automatically:

1) Repo-aware mode
   If the video belongs to the Engro-style project and matching Step-02 / Step-03
   outputs exist, the viewer reuses them:
     - Step-02 scale_config.json -> bar/MAX/MIN locations and OCR settings
     - Step-03 scale_readings.csv -> cleaned per-frame MIN/MAX values

2) Standalone mode
   If the user only has this .py file and a thermal MP4, the script asks them
   once to select:
     - the vertical thermal scale bar
     - the MAX temperature text
     - the MIN temperature text
   It validates those regions with Tesseract OCR, saves a small local config,
   and then performs dynamic OCR in a background thread while playback continues.

Important limitation
--------------------
The input is a rendered MP4, not raw radiometric sensor data. Temperatures are
therefore ESTIMATED from the visible thermal scale/palette, not read directly
from the camera sensor.

Typical command
---------------
    python thermal_hover_video.py "raw-videos\\97. A- 200 Corridor Thermal.mp4"

New/standalone video (Tesseract may be needed):
    python thermal_hover_video.py "C:\\Videos\\new_thermal.mp4" \
        --tesseract "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

Controls
--------
    SPACE       play / pause
    A / D       previous / next frame (pauses)
    J / L       back / forward ~1 second (pauses)
    R           restart from frame 0
    Q / ESC     quit
    Frame slider seek (pauses)

By default the video loops continuously at the end.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Friendly dependency checks.  The file remains standalone, but OpenCV/NumPy
# are still Python packages and Tesseract is required when live OCR is needed.
# -----------------------------------------------------------------------------
try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "OpenCV is required. Install it with:\n"
        "  python -m pip install opencv-python\n"
    ) from exc

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "NumPy is required. Install it with:\n"
        "  python -m pip install numpy\n"
    ) from exc

try:
    import pytesseract
except ImportError:
    pytesseract = None


WINDOW_NAME = "Engro Thermal Hover - Recorded Video"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)

MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 820

OCR_METHODS = [
    ("green_difference", 7),
    ("green_difference", 8),
    ("hsv_green", 7),
    ("green_channel_otsu", 7),
    ("gray_otsu", 7),
    ("gray_adaptive", 7),
]

STEP02_ROOT_NAMES = (
    "step-02-ocr-scale-config",
    "step02-ocr-scale-config",
    "step-2-ocr-scale-config",
)

STEP03_ROOT_NAMES = (
    "step-03-roi-videos",
    "step03-roi-videos",
    "step-3-roi-videos",
)


# =============================================================================
# SMALL UTILITIES
# =============================================================================

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return data


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    os.replace(temporary, path)


def prompt_yes_no(message: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{message} {suffix}: ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in {"y", "yes"}


def point_inside_roi(x: int, y: int, roi: Sequence[int]) -> bool:
    rx, ry, rw, rh = [int(v) for v in roi]
    return rx <= x < rx + rw and ry <= y < ry + rh


def human_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"


# =============================================================================
# PROJECT / CONFIG DISCOVERY
# =============================================================================

def find_named_ancestor(path: Path, name: str) -> Optional[Path]:
    resolved = Path(path).resolve()
    for parent in resolved.parents:
        if parent.name.lower() == name.lower():
            return parent
    return None


def infer_project_root(video_path: Path) -> Optional[Path]:
    raw_root = find_named_ancestor(video_path, "raw-videos")
    return raw_root.parent if raw_root is not None else None


def relative_video_parent(video_path: Path) -> Path:
    raw_root = find_named_ancestor(video_path, "raw-videos")
    if raw_root is None:
        return Path()
    return video_path.resolve().relative_to(raw_root).parent


def standard_stage_file(
    project_root: Path,
    video_path: Path,
    root_names: Sequence[str],
    filename: str,
) -> Optional[Path]:
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


def fallback_unique_stage_file(
    project_root: Path,
    video_stem: str,
    filename: str,
) -> Optional[Path]:
    matches = [
        path.resolve()
        for path in project_root.rglob(filename)
        if path.parent.name == video_stem
    ]
    return matches[0] if len(matches) == 1 else None


def hover_data_dir(video_path: Path, project_root: Optional[Path]) -> Path:
    if project_root is not None:
        return project_root / "thermal-hover" / "data" / video_path.stem
    return video_path.parent / "thermal-hover-data" / video_path.stem


def discover_scale_config(
    video_path: Path,
    project_root: Optional[Path],
    data_dir: Path,
    explicit: Optional[str],
) -> Tuple[Optional[Path], str]:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Scale config not found: {path}")
        return path, "EXPLICIT CONFIG"

    if project_root is not None:
        path = standard_stage_file(
            project_root,
            video_path,
            STEP02_ROOT_NAMES,
            "scale_config.json",
        )
        if path is None:
            path = fallback_unique_stage_file(
                project_root,
                video_path.stem,
                "scale_config.json",
            )
        if path is not None:
            return path, "STEP 02 CONFIG"

    standalone = data_dir / "hover_scale_config.json"
    if standalone.is_file():
        return standalone.resolve(), "THERMAL-HOVER CONFIG"

    return None, "NONE"


def discover_scale_readings(
    video_path: Path,
    project_root: Optional[Path],
    explicit: Optional[str],
) -> Tuple[Optional[Path], str]:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Scale readings CSV not found: {path}")
        return path, "EXPLICIT STEP-03 CSV"

    if project_root is None:
        return None, "NONE"

    path = standard_stage_file(
        project_root,
        video_path,
        STEP03_ROOT_NAMES,
        "scale_readings.csv",
    )
    if path is None:
        path = fallback_unique_stage_file(
            project_root,
            video_path.stem,
            "scale_readings.csv",
        )

    return (path, "STEP 03 CLEAN SCALE") if path is not None else (None, "NONE")


# =============================================================================
# TESSERACT / OCR
# =============================================================================

def configure_tesseract(explicit_path: Optional[str]) -> Optional[str]:
    if pytesseract is None:
        return None

    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"Tesseract executable not found: {candidate}")
        pytesseract.pytesseract.tesseract_cmd = str(candidate)
        return str(candidate)

    if os.name == "nt":
        common = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if common.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(common)
            return str(common)

    return None


def tesseract_available() -> bool:
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def require_tesseract_message() -> str:
    return (
        "Live OCR is required for this video, but Tesseract is unavailable.\n\n"
        "Install the Python wrapper:\n"
        "  python -m pip install pytesseract\n\n"
        "Install Tesseract OCR itself, then run for example:\n"
        "  python thermal_hover_video.py \"VIDEO.mp4\" "
        "--tesseract \"C:\\Program Files\\Tesseract-OCR\\tesseract.exe\"\n\n"
        "If a full Step-03 scale_readings.csv is available, Tesseract is not "
        "required during playback."
    )


def crop_roi(frame: np.ndarray, roi: Sequence[int], padding: int = 0) -> np.ndarray:
    x, y, w, h = [int(v) for v in roi]
    padding = max(0, int(padding))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame.shape[1], x + w + padding)
    y2 = min(frame.shape[0], y + h + padding)
    return frame[y1:y2, x1:x2]


def parse_temperature(text: str, decimal_places: int = 1) -> float:
    if text is None:
        return float("nan")

    cleaned = re.sub(r"[^0-9.\-]", "", str(text).strip())
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return float("nan")

    token = match.group(0)
    try:
        if "." in token:
            return float(token)
        if decimal_places > 0:
            sign = -1.0 if token.startswith("-") else 1.0
            digits = token.lstrip("-")
            # Reject tiny partial OCR such as "1", "8", "15" when one
            # decimal place is expected. This mirrors the safer Step-02 logic.
            if len(digits) < decimal_places + 2:
                return float("nan")
            return sign * int(digits) / (10 ** decimal_places)
        return float(token)
    except ValueError:
        return float("nan")


def finish_binary(binary: np.ndarray) -> Optional[np.ndarray]:
    if binary is None or binary.size == 0:
        return None
    if float(np.mean(binary)) < 127.0:
        binary = cv2.bitwise_not(binary)
    binary = cv2.resize(binary, None, fx=5.0, fy=5.0, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(
        binary, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255
    )


def prepare_ocr_crop(crop: np.ndarray, method: str) -> Optional[np.ndarray]:
    if crop is None or crop.size == 0:
        return None

    b, g, r = cv2.split(crop)

    if method == "green_difference":
        b16 = b.astype(np.int16)
        g16 = g.astype(np.int16)
        r16 = r.astype(np.int16)
        advantage = g16 - np.maximum(r16, b16)
        mask = ((advantage >= 4) & (g >= 70)).astype(np.uint8) * 255
        kernel = np.ones((2, 2), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return finish_binary(mask)

    if method == "hsv_green":
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([25, 12, 65], dtype=np.uint8),
            np.array([100, 255, 255], dtype=np.uint8),
        )
        kernel = np.ones((2, 2), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return finish_binary(mask)

    if method == "green_channel_otsu":
        channel = cv2.GaussianBlur(g, (3, 3), 0)
        _, binary = cv2.threshold(
            channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return finish_binary(binary)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if method == "gray_otsu":
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return finish_binary(binary)

    if method == "gray_adaptive":
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            5,
        )
        return finish_binary(binary)

    raise ValueError(f"Unknown OCR method: {method}")


def run_one_ocr(
    frame: np.ndarray,
    roi: Sequence[int],
    method: str,
    psm: int,
    decimal_places: int,
    padding: int,
) -> dict:
    if pytesseract is None:
        raise RuntimeError(require_tesseract_message())

    crop = crop_roi(frame, roi, padding=padding)
    prepared = prepare_ocr_crop(crop, method)
    if prepared is None:
        return {
            "method": method,
            "psm": int(psm),
            "raw_text": "",
            "value": float("nan"),
        }

    config = f"--psm {int(psm)} -c tessedit_char_whitelist=0123456789.-"
    raw_text = pytesseract.image_to_string(prepared, config=config).strip()
    return {
        "method": method,
        "psm": int(psm),
        "raw_text": raw_text,
        "value": parse_temperature(raw_text, decimal_places),
    }


def candidate_quality(candidate: dict) -> float:
    text = candidate.get("raw_text", "")
    digits = sum(char.isdigit() for char in text)
    return (4.0 if "." in text else 0.0) + min(digits, 4)


def choose_best_pair(
    min_candidates: Sequence[dict],
    max_candidates: Sequence[dict],
    min_allowed: float,
    max_allowed: float,
    min_scale_span: float,
) -> Optional[Tuple[dict, dict]]:
    valid = []
    for min_candidate in min_candidates:
        min_value = min_candidate["value"]
        if not np.isfinite(min_value) or not min_allowed <= min_value <= max_allowed:
            continue
        for max_candidate in max_candidates:
            max_value = max_candidate["value"]
            if not np.isfinite(max_value) or not min_allowed <= max_value <= max_allowed:
                continue
            if max_value - min_value < min_scale_span:
                continue
            score = candidate_quality(min_candidate) + candidate_quality(max_candidate)
            if (
                min_candidate["method"] == max_candidate["method"]
                and min_candidate["psm"] == max_candidate["psm"]
            ):
                score += 1.0
            valid.append((score, min_candidate, max_candidate))

    if not valid:
        return None
    valid.sort(key=lambda item: item[0], reverse=True)
    _, best_min, best_max = valid[0]
    return best_min, best_max


def ordered_ocr_methods(scale: dict) -> List[Tuple[str, int]]:
    preferred = [
        (scale.get("min_ocr_method", "green_difference"), int(scale.get("min_ocr_psm", 7))),
        (scale.get("max_ocr_method", "green_difference"), int(scale.get("max_ocr_psm", 7))),
    ]
    output = []
    seen = set()
    for item in [*preferred, *OCR_METHODS]:
        key = (str(item[0]), int(item[1]))
        if key not in seen:
            seen.add(key)
            output.append(key)
    return output


def ocr_scale_frame(
    frame: np.ndarray,
    scale: dict,
    min_allowed: float,
    max_allowed: float,
    min_scale_span: float,
    use_all_methods: bool = True,
) -> Optional[dict]:
    decimal_places = int(scale.get("display_decimal_places", 1))
    padding = int(scale.get("ocr_padding_pixels", 6))

    if use_all_methods:
        methods = ordered_ocr_methods(scale)
    else:
        methods = [
            (scale.get("min_ocr_method", "green_difference"), int(scale.get("min_ocr_psm", 7))),
            (scale.get("max_ocr_method", "green_difference"), int(scale.get("max_ocr_psm", 7))),
        ]

    min_candidates = []
    max_candidates = []
    for method, psm in methods:
        min_candidates.append(
            run_one_ocr(
                frame,
                scale["min_text_roi"],
                method,
                psm,
                decimal_places,
                padding,
            )
        )
        max_candidates.append(
            run_one_ocr(
                frame,
                scale["max_text_roi"],
                method,
                psm,
                decimal_places,
                padding,
            )
        )

    pair = choose_best_pair(
        min_candidates,
        max_candidates,
        min_allowed,
        max_allowed,
        min_scale_span,
    )
    if pair is None:
        return None

    best_min, best_max = pair
    return {
        "min_temp": float(best_min["value"]),
        "max_temp": float(best_max["value"]),
        "min_text": best_min["raw_text"],
        "max_text": best_max["raw_text"],
        "min_method": best_min["method"],
        "min_psm": int(best_min["psm"]),
        "max_method": best_max["method"],
        "max_psm": int(best_max["psm"]),
    }


def ocr_scale_primary_then_retry(
    frame: np.ndarray,
    scale: dict,
    min_allowed: float,
    max_allowed: float,
    min_scale_span: float,
) -> Optional[dict]:
    """Fast normal path for continuous playback.

    First use only the MIN and MAX OCR methods already chosen during setup
    (normally just two Tesseract calls total). Only if that pair is unusable
    do we spend time trying the full fallback method set. This keeps playback
    responsive while preserving the robust recovery behaviour.
    """
    decimal_places = int(scale.get("display_decimal_places", 1))
    padding = int(scale.get("ocr_padding_pixels", 6))

    min_candidate = run_one_ocr(
        frame,
        scale["min_text_roi"],
        scale.get("min_ocr_method", "green_difference"),
        int(scale.get("min_ocr_psm", 7)),
        decimal_places,
        padding,
    )
    max_candidate = run_one_ocr(
        frame,
        scale["max_text_roi"],
        scale.get("max_ocr_method", "green_difference"),
        int(scale.get("max_ocr_psm", 7)),
        decimal_places,
        padding,
    )

    pair = choose_best_pair(
        [min_candidate],
        [max_candidate],
        min_allowed,
        max_allowed,
        min_scale_span,
    )
    if pair is not None:
        best_min, best_max = pair
        return {
            "min_temp": float(best_min["value"]),
            "max_temp": float(best_max["value"]),
            "min_text": best_min["raw_text"],
            "max_text": best_max["raw_text"],
            "min_method": best_min["method"],
            "min_psm": int(best_min["psm"]),
            "max_method": best_max["method"],
            "max_psm": int(best_max["psm"]),
        }

    return ocr_scale_frame(
        frame,
        scale,
        min_allowed,
        max_allowed,
        min_scale_span,
        use_all_methods=True,
    )


# =============================================================================
# FIRST-TIME SCALE ANNOTATION (standalone mode)
# =============================================================================

def selection_display(frame: np.ndarray) -> Tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height, 1.0)
    if scale >= 0.999:
        return frame.copy(), 1.0
    resized = cv2.resize(
        frame,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def select_roi_scaled(title: str, frame: np.ndarray) -> List[int]:
    display, scale = selection_display(frame)
    roi = cv2.selectROI(title, display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise RuntimeError(f"No valid ROI selected for {title}")
    if scale <= 0:
        scale = 1.0
    original = [
        int(round(x / scale)),
        int(round(y / scale)),
        int(round(w / scale)),
        int(round(h / scale)),
    ]
    original[0] = max(0, min(original[0], frame.shape[1] - 1))
    original[1] = max(0, min(original[1], frame.shape[0] - 1))
    original[2] = max(1, min(original[2], frame.shape[1] - original[0]))
    original[3] = max(1, min(original[3], frame.shape[0] - original[1]))
    return original


def draw_roi_preview(frame: np.ndarray, scale: dict) -> np.ndarray:
    preview = frame.copy()
    items = [
        (scale["bar_roi"], "THERMAL BAR", CYAN),
        (scale["max_text_roi"], "MAX", RED),
        (scale["min_text_roi"], "MIN", GREEN),
    ]
    for roi, label, color in items:
        x, y, w, h = [int(v) for v in roi]
        cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            preview,
            label,
            (x, max(24, y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return preview


def create_standalone_scale_config(
    video_path: Path,
    setup_frame: np.ndarray,
    setup_frame_number: int,
    width: int,
    height: int,
    fps: float,
    total_frames: int,
    data_dir: Path,
    args,
) -> Path:
    if not tesseract_available():
        raise RuntimeError(require_tesseract_message())

    print("\n============================================================")
    print("FIRST-TIME THERMAL SCALE SETUP")
    print("============================================================")
    print("This video has no reusable thermal-scale configuration.")
    print("You only need to define the camera UI regions; Step 01 polygons are NOT needed.")
    print("For each window: drag the rectangle, then press ENTER or SPACE.")

    while True:
        print("\n1/3 Select ONLY the vertical thermal scale/palette bar.")
        bar_roi = select_roi_scaled("1/3 - Select THERMAL SCALE BAR", setup_frame)

        print("2/3 Select the displayed MAX temperature number.")
        max_roi = select_roi_scaled("2/3 - Select MAX TEMPERATURE TEXT", setup_frame)

        print("3/3 Select the displayed MIN temperature number.")
        min_roi = select_roi_scaled("3/3 - Select MIN TEMPERATURE TEXT", setup_frame)

        provisional = {
            "bar_roi": bar_roi,
            "max_text_roi": max_roi,
            "min_text_roi": min_roi,
            "bar_horizontal_inner_fraction": 0.20,
            "display_decimal_places": int(args.decimal_places),
            "ocr_padding_pixels": int(args.ocr_padding),
            # Defaults used as preferred methods until the validation pass picks better ones.
            "min_ocr_method": "green_difference",
            "min_ocr_psm": 7,
            "max_ocr_method": "green_difference",
            "max_ocr_psm": 7,
        }

        print("\nValidating selected MAX/MIN regions with OCR...")
        result = ocr_scale_frame(
            setup_frame,
            provisional,
            args.min_allowed_temp,
            args.max_allowed_temp,
            args.min_scale_span,
            use_all_methods=True,
        )

        if result is None:
            print("\nCould not read a plausible MIN/MAX pair from those selections.")
            if prompt_yes_no("Redraw the three scale regions?", default=True):
                continue
            raise RuntimeError("Scale configuration cancelled because OCR validation failed.")

        print(
            f"\nOCR validation succeeded:\n"
            f"  MIN: {result['min_temp']:.1f} C from {result['min_text']!r} "
            f"using {result['min_method']} / PSM {result['min_psm']}\n"
            f"  MAX: {result['max_temp']:.1f} C from {result['max_text']!r} "
            f"using {result['max_method']} / PSM {result['max_psm']}\n"
            f"  Span: {result['max_temp'] - result['min_temp']:.1f} C"
        )

        if not prompt_yes_no("Use this scale setup?", default=True):
            continue

        provisional.update(
            {
                "min_ocr_method": result["min_method"],
                "min_ocr_psm": result["min_psm"],
                "max_ocr_method": result["max_method"],
                "max_ocr_psm": result["max_psm"],
                "reference_ocr_min_temp": result["min_temp"],
                "reference_ocr_max_temp": result["max_temp"],
                "reference_ocr_min_text": result["min_text"],
                "reference_ocr_max_text": result["max_text"],
                "min_scale_span_c": float(args.min_scale_span),
            }
        )

        config = {
            "format": "thermal_hover_scale_config_v1",
            "video": {
                "filename": video_path.name,
                "width": int(width),
                "height": int(height),
                "fps": float(fps),
                "total_frames": int(total_frames),
                "reference_frame_number": int(setup_frame_number),
            },
            "scale": provisional,
        }

        data_dir.mkdir(parents=True, exist_ok=True)
        config_path = data_dir / "hover_scale_config.json"
        save_json(config_path, config)

        preview = draw_roi_preview(setup_frame, provisional)
        cv2.imwrite(str(data_dir / "hover_scale_config_preview.png"), preview)

        print(f"\nSaved reusable hover configuration:\n  {config_path}")
        return config_path.resolve()


# =============================================================================
# STEP-03 CLEAN SCALE CACHE
# =============================================================================

def load_scale_readings_csv(path: Path, total_frames: int):
    mins = np.full(total_frames, np.nan, dtype=np.float32)
    maxs = np.full(total_frames, np.nan, dtype=np.float32)
    loaded = 0

    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        required = {"frame", "clean_min_temp_used", "clean_max_temp_used"}
        if not required.issubset(fields):
            raise RuntimeError(
                "scale_readings.csv is missing expected Step-03 columns: "
                "frame, clean_min_temp_used, clean_max_temp_used"
            )

        for row in reader:
            try:
                frame_number = int(row["frame"])
                min_temp = float(row["clean_min_temp_used"])
                max_temp = float(row["clean_max_temp_used"])
            except (TypeError, ValueError):
                continue
            if not 0 <= frame_number < total_frames:
                continue
            if not (np.isfinite(min_temp) and np.isfinite(max_temp) and max_temp > min_temp):
                continue
            mins[frame_number] = min_temp
            maxs[frame_number] = max_temp
            loaded += 1

    return mins, maxs, loaded


# =============================================================================
# COLOR BAR -> TEMPERATURE LUT
# =============================================================================

def build_temperature_lut(
    frame: np.ndarray,
    bar_roi: Sequence[int],
    min_temp: float,
    max_temp: float,
    inner_fraction: float = 0.20,
) -> np.ndarray:
    bar = crop_roi(frame, bar_roi, padding=0)
    if bar is None or bar.size == 0:
        raise RuntimeError("Thermal scale-bar crop is empty.")

    bar_h, bar_w = bar.shape[:2]
    left = int(round(bar_w * inner_fraction))
    right = int(round(bar_w * (1.0 - inner_fraction)))
    left = max(0, min(left, bar_w - 1))
    right = max(left + 1, min(right, bar_w))

    bar_gray = cv2.cvtColor(bar[:, left:right], cv2.COLOR_BGR2GRAY)
    palette_intensity = np.median(bar_gray, axis=1).astype(np.float32)
    palette_temperature = np.linspace(
        float(max_temp), float(min_temp), num=bar_h, dtype=np.float32
    )

    levels = np.arange(256, dtype=np.float32)[:, None]
    nearest_row = np.argmin(
        np.abs(levels - palette_intensity[None, :]), axis=1
    )
    return palette_temperature[nearest_row]


def sample_hover_3x3(
    frame: np.ndarray,
    lut: np.ndarray,
    x: int,
    y: int,
    max_channel_difference: int,
    scale: dict,
) -> dict:
    height, width = frame.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return {"valid": False, "reason": "OUTSIDE FRAME"}

    # The scale bar and its labels are camera UI, not a plant surface.
    for roi in (scale["bar_roi"], scale["max_text_roi"], scale["min_text_roi"]):
        if point_inside_roi(x, y, roi):
            return {"valid": False, "reason": "THERMAL SCALE / UI REGION"}

    x1 = max(0, x - 1)
    x2 = min(width, x + 2)
    y1 = max(0, y - 1)
    y2 = min(height, y + 2)

    patch = frame[y1:y2, x1:x2]
    patch_i16 = patch.astype(np.int16)
    spread = patch_i16.max(axis=2) - patch_i16.min(axis=2)
    valid = spread <= int(max_channel_difference)

    if not np.any(valid):
        return {"valid": False, "reason": "COLORED UI / OVERLAY"}

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    temperatures = lut[gray][valid]
    if temperatures.size == 0:
        return {"valid": False, "reason": "NO VALID THERMAL PIXELS"}

    return {
        "valid": True,
        "temperature": float(np.mean(temperatures)),
        "minimum": float(np.min(temperatures)),
        "maximum": float(np.max(temperatures)),
        "sample_count": int(temperatures.size),
    }


# =============================================================================
# BACKGROUND DYNAMIC OCR FOR NEW / PARTIALLY PROCESSED VIDEOS
# =============================================================================

class AsyncScaleOCR:
    def __init__(self, scale: dict, fps: float, args):
        self.scale = scale
        self.fps = float(fps)
        self.args = args
        self.interval_frames = max(1, int(round(self.fps / float(args.ocr_hz))))
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="thermal-ocr")
        self.future = None
        self.future_frame = None
        self.last_submit_frame = -10**9
        self.trusted_min = self._finite_or_none(scale.get("reference_ocr_min_temp"))
        self.trusted_max = self._finite_or_none(scale.get("reference_ocr_max_temp"))
        self.status = "REFERENCE SCALE" if self.trusted_min is not None and self.trusted_max is not None else "WAITING FOR OCR"
        self.last_accepted_frame = None
        self.last_ocr_message = ""

    @staticmethod
    def _finite_or_none(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)

    def reset_to_reference(self):
        self.trusted_min = self._finite_or_none(self.scale.get("reference_ocr_min_temp"))
        self.trusted_max = self._finite_or_none(self.scale.get("reference_ocr_max_temp"))
        self.status = "REFERENCE SCALE" if self.trusted_min is not None and self.trusted_max is not None else "WAITING FOR OCR"
        self.last_submit_frame = -10**9
        self.last_accepted_frame = None

    def _worker(self, frame: np.ndarray):
        return ocr_scale_primary_then_retry(
            frame,
            self.scale,
            self.args.min_allowed_temp,
            self.args.max_allowed_temp,
            self.args.min_scale_span,
        )

    def _accept_candidate(self, frame_number: int, result: Optional[dict], allow_jump: bool = False):
        if result is None:
            self.status = "OCR FAILED - HOLD PREVIOUS"
            self.last_ocr_message = "No plausible MIN/MAX pair"
            return False

        candidate_min = float(result["min_temp"])
        candidate_max = float(result["max_temp"])

        if (
            not allow_jump
            and self.trusted_min is not None
            and self.trusted_max is not None
            and float(self.args.max_scale_jump) > 0
        ):
            if (
                abs(candidate_min - self.trusted_min) > float(self.args.max_scale_jump)
                or abs(candidate_max - self.trusted_max) > float(self.args.max_scale_jump)
            ):
                self.status = "OCR JUMP REJECTED - HOLD PREVIOUS"
                self.last_ocr_message = (
                    f"candidate {candidate_min:.1f}..{candidate_max:.1f} C rejected"
                )
                return False

        self.trusted_min = candidate_min
        self.trusted_max = candidate_max
        self.last_accepted_frame = int(frame_number)
        self.status = "LIVE OCR"
        self.last_ocr_message = (
            f"{candidate_min:.1f}..{candidate_max:.1f} C"
        )
        return True

    def poll(self):
        if self.future is None or not self.future.done():
            return
        future = self.future
        frame_number = self.future_frame
        self.future = None
        self.future_frame = None
        try:
            result = future.result()
        except Exception as exc:
            self.status = "OCR ERROR - HOLD PREVIOUS"
            self.last_ocr_message = str(exc)
            return
        self._accept_candidate(frame_number, result, allow_jump=False)

    def maybe_submit(self, frame_number: int, frame: np.ndarray):
        self.poll()
        if self.future is not None:
            return
        if frame_number - self.last_submit_frame < self.interval_frames:
            return
        self.last_submit_frame = int(frame_number)
        self.future_frame = int(frame_number)
        self.future = self.executor.submit(self._worker, frame.copy())

    def synchronous_reseed(self, frame_number: int, frame: np.ndarray):
        """Used after a large manual seek where temporal jump validation is inappropriate."""
        result = self._worker(frame.copy())
        accepted = self._accept_candidate(frame_number, result, allow_jump=True)
        self.last_submit_frame = int(frame_number)
        return accepted

    def current(self) -> Optional[Tuple[float, float]]:
        self.poll()
        if self.trusted_min is None or self.trusted_max is None:
            return None
        return float(self.trusted_min), float(self.trusted_max)


# =============================================================================
# DRAWING
# =============================================================================

def draw_status_box(image: np.ndarray, lines: Sequence[str], color=WHITE) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    sizes = [cv2.getTextSize(str(line), font, font_scale, thickness) for line in lines]
    width = max((size[0][0] for size in sizes), default=200) + 22
    line_height = max((size[0][1] + size[1] for size in sizes), default=17) + 7
    height = line_height * len(lines) + 10
    cv2.rectangle(image, (10, 10), (min(image.shape[1] - 1, 10 + width), min(image.shape[0] - 1, 10 + height)), BLACK, -1)
    y = 10 + line_height
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            str(line),
            (20, y),
            font,
            font_scale,
            color if index == 0 else WHITE,
            thickness,
            cv2.LINE_AA,
        )
        y += line_height


def draw_hover_tooltip(image: np.ndarray, x: int, y: int, lines: Sequence[str], valid=True) -> None:
    color = GREEN if valid else RED
    cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 20, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.53
    thickness = 1
    line_height = 21
    widths = [cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines]
    box_w = max(widths, default=120) + 18
    box_h = len(lines) * line_height + 10

    x1, y1 = x + 15, y + 15
    if x1 + box_w >= image.shape[1]:
        x1 = max(0, x - box_w - 15)
    if y1 + box_h >= image.shape[0]:
        y1 = max(0, y - box_h - 15)

    x2 = min(image.shape[1] - 1, x1 + box_w)
    y2 = min(image.shape[0] - 1, y1 + box_h)
    cv2.rectangle(image, (x1, y1), (x2, y2), BLACK, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    text_y = y1 + 19
    for line in lines:
        cv2.putText(image, line, (x1 + 8, text_y), font, font_scale, WHITE, thickness, cv2.LINE_AA)
        text_y += line_height


def draw_controls(image: np.ndarray) -> None:
    text = "SPACE Play/Pause | A/D +/-1 frame | J/L +/-1 sec | R Restart | Q/ESC Quit | Slider Seek"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    y1 = max(0, image.shape[0] - h - baseline - 12)
    cv2.rectangle(image, (0, y1), (min(image.shape[1] - 1, w + 18), image.shape[0] - 1), BLACK, -1)
    cv2.putText(image, text, (8, image.shape[0] - 7), font, scale, WHITE, thickness, cv2.LINE_AA)


# =============================================================================
# VIDEO PLAYER
# =============================================================================

class ThermalHoverVideoPlayer:
    def __init__(
        self,
        video_path: Path,
        scale_config: dict,
        scale_config_source: str,
        scale_readings_path: Optional[Path],
        scale_readings_source: str,
        args,
    ):
        self.video_path = video_path.resolve()
        self.scale_config = scale_config
        self.scale = scale_config["scale"]
        self.scale_config_source = scale_config_source
        self.scale_readings_path = scale_readings_path
        self.scale_readings_source = scale_readings_source
        self.args = args

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps <= 0 or self.total_frames <= 0:
            self.cap.release()
            raise RuntimeError("Video reported invalid FPS/frame count.")

        cfg_video = scale_config.get("video", {})
        if int(cfg_video.get("width", self.width)) != self.width or int(cfg_video.get("height", self.height)) != self.height:
            self.cap.release()
            raise RuntimeError(
                "Scale configuration resolution does not match this video.\n"
                f"Video : {self.width} x {self.height}\n"
                f"Config: {cfg_video.get('width')} x {cfg_video.get('height')}"
            )

        self.cached_min = np.full(self.total_frames, np.nan, dtype=np.float32)
        self.cached_max = np.full(self.total_frames, np.nan, dtype=np.float32)
        self.cached_rows = 0
        if scale_readings_path is not None:
            self.cached_min, self.cached_max, self.cached_rows = load_scale_readings_csv(
                scale_readings_path, self.total_frames
            )

        # Live OCR is only needed for frames with no Step-03 clean value.
        self.ocr_tracker = None
        cache_complete = self.cached_rows >= self.total_frames
        if not cache_complete:
            if tesseract_available():
                self.ocr_tracker = AsyncScaleOCR(self.scale, self.fps, args)
            elif self.cached_rows == 0:
                self.cap.release()
                raise RuntimeError(require_tesseract_message())

        ok, first_frame = self.cap.read()
        if not ok:
            self.cap.release()
            raise RuntimeError("Could not read first video frame.")

        self.frame_index = 0
        self.current_frame = first_frame
        self.playing = not bool(args.start_paused)
        self.loop_count = 0
        self.next_deadline = time.perf_counter()

        self.mouse_display_x = None
        self.mouse_display_y = None
        self.display_width = None
        self.display_height = None

        self.current_min = None
        self.current_max = None
        self.current_scale_source = ""
        self.current_lut = None
        self.pending_seek = None
        self.updating_trackbar = False

        # Seed current scale immediately.
        self._update_current_scale_and_lut(force_ocr_if_needed=True)

    def close(self):
        if self.ocr_tracker is not None:
            self.ocr_tracker.close()
        self.cap.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # SCALE FOR CURRENT FRAME
    # ------------------------------------------------------------------
    def _cached_scale(self, frame_number: int) -> Optional[Tuple[float, float]]:
        if not 0 <= frame_number < self.total_frames:
            return None
        min_temp = float(self.cached_min[frame_number])
        max_temp = float(self.cached_max[frame_number])
        if np.isfinite(min_temp) and np.isfinite(max_temp) and max_temp > min_temp:
            return min_temp, max_temp
        return None

    def _update_current_scale_and_lut(self, force_ocr_if_needed=False):
        cached = self._cached_scale(self.frame_index)
        if cached is not None:
            self.current_min, self.current_max = cached
            self.current_scale_source = "STEP 03 CLEAN SCALE"
        elif self.ocr_tracker is not None:
            if force_ocr_if_needed and self.ocr_tracker.current() is None:
                print(f"\nReading thermal scale at frame {self.frame_index}...")
                self.ocr_tracker.synchronous_reseed(self.frame_index, self.current_frame)
            else:
                self.ocr_tracker.maybe_submit(self.frame_index, self.current_frame)
            current = self.ocr_tracker.current()
            if current is not None:
                self.current_min, self.current_max = current
                self.current_scale_source = self.ocr_tracker.status
            else:
                self.current_min = None
                self.current_max = None
                self.current_scale_source = self.ocr_tracker.status
        else:
            self.current_min = None
            self.current_max = None
            self.current_scale_source = "NO SCALE AVAILABLE"

        self.current_lut = None
        if self.current_min is not None and self.current_max is not None:
            try:
                self.current_lut = build_temperature_lut(
                    self.current_frame,
                    self.scale["bar_roi"],
                    self.current_min,
                    self.current_max,
                    float(self.scale.get("bar_horizontal_inner_fraction", 0.20)),
                )
            except Exception:
                self.current_lut = None
                self.current_scale_source = "BAR READ ERROR"

    # ------------------------------------------------------------------
    # VIDEO POSITIONING
    # ------------------------------------------------------------------
    def _load_frame(self, frame_number: int, pause=True, reseed_ocr=True):
        frame_number = int(np.clip(frame_number, 0, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_number}")
        self.frame_index = frame_number
        self.current_frame = frame
        if pause:
            self.playing = False
        self.next_deadline = time.perf_counter()

        if reseed_ocr and self._cached_scale(frame_number) is None and self.ocr_tracker is not None:
            print(f"\nRe-reading scale after seek to frame {frame_number}...")
            try:
                self.ocr_tracker.synchronous_reseed(frame_number, frame)
            except Exception as exc:
                print(f"OCR seek reseed warning: {exc}")
        self._update_current_scale_and_lut(force_ocr_if_needed=False)

    def _restart_for_loop(self):
        self.loop_count += 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.cap.read()
        if not ok:
            self.playing = False
            return
        self.frame_index = 0
        self.current_frame = frame
        if self.ocr_tracker is not None:
            self.ocr_tracker.reset_to_reference()
        self._update_current_scale_and_lut(force_ocr_if_needed=False)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        self.next_deadline = time.perf_counter() + 1.0 / self.fps

    def _read_next(self):
        if self.frame_index >= self.total_frames - 1:
            if self.args.loop:
                self._restart_for_loop()
                return
            self.playing = False
            return

        ok, frame = self.cap.read()
        if not ok:
            if self.args.loop:
                self._restart_for_loop()
            else:
                self.playing = False
            return

        self.frame_index += 1
        self.current_frame = frame
        self._update_current_scale_and_lut(force_ocr_if_needed=False)

    def _toggle_play(self):
        if self.playing:
            self.playing = False
            return
        if self.frame_index >= self.total_frames - 1:
            if self.args.loop:
                self._restart_for_loop()
            else:
                self._load_frame(0, pause=False, reseed_ocr=True)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index + 1)
        self.playing = True
        self.next_deadline = time.perf_counter()

    def _advance_if_due(self):
        if not self.playing:
            # Still poll background OCR so a result can arrive while paused.
            if self.ocr_tracker is not None and self._cached_scale(self.frame_index) is None:
                before = self.ocr_tracker.status
                self.ocr_tracker.poll()
                if self.ocr_tracker.status != before:
                    self._update_current_scale_and_lut(force_ocr_if_needed=False)
            return

        now = time.perf_counter()
        if now < self.next_deadline:
            if self.ocr_tracker is not None:
                self.ocr_tracker.poll()
            return

        self._read_next()
        period = 1.0 / self.fps
        self.next_deadline += period
        if now - self.next_deadline > 0.5:
            self.next_deadline = now + period

    # ------------------------------------------------------------------
    # MOUSE / DISPLAY
    # ------------------------------------------------------------------
    def _display_size(self):
        scale = min(
            MAX_DISPLAY_WIDTH / self.width,
            MAX_DISPLAY_HEIGHT / self.height,
            1.0,
        )
        return (
            max(1, int(round(self.width * scale))),
            max(1, int(round(self.height * scale))),
        )

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_MOUSEMOVE:
            return
        self.mouse_display_x = int(x)
        self.mouse_display_y = int(y)

    def _mouse_original(self) -> Optional[Tuple[int, int]]:
        if (
            self.mouse_display_x is None
            or self.mouse_display_y is None
            or self.display_width is None
            or self.display_height is None
        ):
            return None
        if not (
            0 <= self.mouse_display_x < self.display_width
            and 0 <= self.mouse_display_y < self.display_height
        ):
            return None
        x = int(round(self.mouse_display_x * self.width / self.display_width))
        y = int(round(self.mouse_display_y * self.height / self.display_height))
        return (
            max(0, min(x, self.width - 1)),
            max(0, min(y, self.height - 1)),
        )

    def _render(self) -> np.ndarray:
        self.display_width, self.display_height = self._display_size()
        display = cv2.resize(
            self.current_frame,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA if self.display_width < self.width else cv2.INTER_LINEAR,
        )

        state = "PLAYING - HOVER ACTIVE" if self.playing else "PAUSED - HOVER ACTIVE"
        state_color = YELLOW if self.playing else GREEN
        timestamp = self.frame_index / self.fps
        duration = self.total_frames / self.fps

        status_lines = [
            state,
            f"Frame {self.frame_index:,}/{self.total_frames - 1:,} | {human_time(timestamp)} / {human_time(duration)} | loop {self.loop_count}",
        ]
        if self.current_min is not None and self.current_max is not None:
            status_lines.append(
                f"Scale {self.current_min:.1f}..{self.current_max:.1f} C | {self.current_scale_source}"
            )
        else:
            status_lines.append(f"Scale unavailable | {self.current_scale_source}")

        draw_status_box(display, status_lines, color=state_color)

        mouse = self._mouse_original()
        if mouse is not None:
            original_x, original_y = mouse
            if self.current_lut is None:
                draw_hover_tooltip(
                    display,
                    self.mouse_display_x,
                    self.mouse_display_y,
                    [f"Pixel x={original_x}, y={original_y}", "Temperature unavailable"],
                    valid=False,
                )
            else:
                result = sample_hover_3x3(
                    self.current_frame,
                    self.current_lut,
                    original_x,
                    original_y,
                    self.args.max_channel_difference,
                    self.scale,
                )
                if result["valid"]:
                    draw_hover_tooltip(
                        display,
                        self.mouse_display_x,
                        self.mouse_display_y,
                        [
                            f"3x3 avg: {result['temperature']:.1f} C",
                            f"Pixel x={original_x}, y={original_y}",
                            f"Valid samples: {result['sample_count']}/9",
                        ],
                        valid=True,
                    )
                else:
                    draw_hover_tooltip(
                        display,
                        self.mouse_display_x,
                        self.mouse_display_y,
                        [f"Pixel x={original_x}, y={original_y}", result["reason"], "Temperature: N/A"],
                        valid=False,
                    )

        draw_controls(display)
        return display

    # ------------------------------------------------------------------
    # TRACKBAR / MAIN LOOP
    # ------------------------------------------------------------------
    def _trackbar_callback(self, position):
        if not self.updating_trackbar:
            self.pending_seek = int(position)

    def _update_trackbar(self):
        self.updating_trackbar = True
        cv2.setTrackbarPos("Frame", WINDOW_NAME, int(self.frame_index))
        self.updating_trackbar = False

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar(
            "Frame",
            WINDOW_NAME,
            0,
            max(1, self.total_frames - 1),
            self._trackbar_callback,
        )
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        print("\n============================================================")
        print("THERMAL HOVER VIDEO")
        print("============================================================")
        print(f"Video          : {self.video_path}")
        print(f"Resolution     : {self.width} x {self.height}")
        print(f"FPS            : {self.fps:.3f}")
        print(f"Frames         : {self.total_frames}")
        print(f"Scale config   : {self.scale_config_source}")
        print(f"Step-03 scale  : {self.scale_readings_source}")
        if self.scale_readings_path is not None:
            print(f"Cached frames  : {self.cached_rows}/{self.total_frames}")
        if self.ocr_tracker is not None:
            print(f"Live OCR       : enabled at ~{self.args.ocr_hz:.2f} Hz in background")
        else:
            print("Live OCR       : not needed")
        print(f"Auto-loop      : {'ON' if self.args.loop else 'OFF'}")
        print("Hover          : ACTIVE during playback and pause (3x3 average)")
        print("\nControls: SPACE play/pause | A/D frame | J/L second | R restart | Q/ESC quit")

        try:
            while True:
                if self.pending_seek is not None:
                    requested = self.pending_seek
                    self.pending_seek = None
                    if requested != self.frame_index:
                        self._load_frame(requested, pause=True, reseed_ocr=True)

                self._advance_if_due()
                display = self._render()
                cv2.imshow(WINDOW_NAME, display)
                self._update_trackbar()

                key = cv2.waitKeyEx(5)
                if key == -1:
                    continue
                if key in (27, ord("q"), ord("Q")):
                    break
                if key == 32:
                    self._toggle_play()
                    continue
                if key in (ord("a"), ord("A")):
                    self._load_frame(self.frame_index - 1, pause=True, reseed_ocr=True)
                    continue
                if key in (ord("d"), ord("D")):
                    self._load_frame(self.frame_index + 1, pause=True, reseed_ocr=True)
                    continue
                if key in (ord("j"), ord("J")):
                    self._load_frame(self.frame_index - int(round(self.fps)), pause=True, reseed_ocr=True)
                    continue
                if key in (ord("l"), ord("L")):
                    self._load_frame(self.frame_index + int(round(self.fps)), pause=True, reseed_ocr=True)
                    continue
                if key in (ord("r"), ord("R")):
                    self._load_frame(0, pause=not self.playing, reseed_ocr=True)
                    continue
        finally:
            self.close()


# =============================================================================
# VALIDATION / MAIN
# =============================================================================

def read_video_metadata(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Video reported invalid metadata.")
    return fps, width, height, total_frames


def read_specific_frame(video_path: Path, frame_number: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read setup frame {frame_number}")
    return frame


def validate_scale_config(config: dict, width: int, height: int) -> None:
    if "scale" not in config or not isinstance(config["scale"], dict):
        raise RuntimeError("Scale config has no 'scale' object.")
    scale = config["scale"]
    required = {"bar_roi", "max_text_roi", "min_text_roi"}
    missing = sorted(required - set(scale))
    if missing:
        raise RuntimeError(f"Scale config is missing fields: {missing}")

    cfg_video = config.get("video", {})
    cfg_width = int(cfg_video.get("width", width))
    cfg_height = int(cfg_video.get("height", height))
    if (cfg_width, cfg_height) != (width, height):
        raise RuntimeError(
            "Scale config belongs to a different resolution.\n"
            f"Video : {width} x {height}\nConfig: {cfg_width} x {cfg_height}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Play a thermal MP4 continuously and show estimated 3x3 hover "
            "temperature during playback or pause. Works with Engro Step-02/03 "
            "outputs when present, or configures a completely new video itself."
        )
    )
    parser.add_argument("video", help="Path to rendered thermal MP4")
    parser.add_argument("--scale-config", default=None, help="Optional explicit scale_config.json")
    parser.add_argument("--scale-readings", default=None, help="Optional explicit Step-03 scale_readings.csv")
    parser.add_argument("--tesseract", default=None, help="Optional path to tesseract.exe")
    parser.add_argument("--ocr-hz", type=float, default=3.0, help="Background OCR frequency when Step-03 scale is unavailable. Default: 3")
    parser.add_argument("--max-scale-jump", type=float, default=2.5, help="Reject sudden live OCR MIN/MAX jumps larger than this many C. Default: 2.5")
    parser.add_argument("--max-channel-difference", type=int, default=18, help="Reject strongly colored UI pixels from 3x3 hover sample. Default: 18")
    parser.add_argument("--min-allowed-temp", type=float, default=-100.0)
    parser.add_argument("--max-allowed-temp", type=float, default=1000.0)
    parser.add_argument("--min-scale-span", type=float, default=5.0)
    parser.add_argument("--decimal-places", type=int, default=1, help="Expected displayed decimal places during first-time setup. Default: 1")
    parser.add_argument("--ocr-padding", type=int, default=6, help="Extra pixels around selected MIN/MAX text regions. Default: 6")
    parser.add_argument("--setup-frame", type=int, default=0, help="Frame used for first-time scale annotation. Default: 0")
    parser.add_argument("--start-paused", action="store_true", help="Open paused instead of playing immediately")
    parser.add_argument("--no-loop", dest="loop", action="store_false", help="Do not automatically loop at the end")
    parser.set_defaults(loop=True)
    return parser


def main():
    args = build_parser().parse_args()

    if args.ocr_hz <= 0:
        raise SystemExit("--ocr-hz must be > 0")
    if args.min_scale_span <= 0:
        raise SystemExit("--min-scale-span must be > 0")
    if args.max_channel_difference < 0:
        raise SystemExit("--max-channel-difference must be >= 0")

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fps, width, height, total_frames = read_video_metadata(video_path)
    project_root = infer_project_root(video_path)
    data_dir = hover_data_dir(video_path, project_root)

    configured_tesseract = configure_tesseract(args.tesseract)

    config_path, config_source = discover_scale_config(
        video_path, project_root, data_dir, args.scale_config
    )
    readings_path, readings_source = discover_scale_readings(
        video_path, project_root, args.scale_readings
    )

    # If no reusable config exists, make this code independently usable by
    # creating the scale config itself. Step 01/02/03 are not prerequisites.
    if config_path is None:
        setup_frame_number = max(0, min(int(args.setup_frame), total_frames - 1))
        setup_frame = read_specific_frame(video_path, setup_frame_number)
        config_path = create_standalone_scale_config(
            video_path,
            setup_frame,
            setup_frame_number,
            width,
            height,
            fps,
            total_frames,
            data_dir,
            args,
        )
        config_source = "THERMAL-HOVER FIRST-TIME CONFIG"

    config = load_json(config_path)
    validate_scale_config(config, width, height)

    # If the selected config has its own minimum scale span, preserve it unless
    # the CLI explicitly used another value (the CLI default remains sensible).
    config_span = config.get("scale", {}).get("min_scale_span_c")
    if config_span is not None and math.isclose(args.min_scale_span, 5.0):
        try:
            args.min_scale_span = float(config_span)
        except (TypeError, ValueError):
            pass

    print("\nInput preparation:")
    print(f"  Video        : {video_path}")
    print(f"  Project mode : {'YES' if project_root is not None else 'NO / STANDALONE'}")
    print(f"  Config       : {config_source} -> {config_path}")
    print(f"  Step-03 CSV  : {readings_source}" + (f" -> {readings_path}" if readings_path else ""))
    if configured_tesseract:
        print(f"  Tesseract    : {configured_tesseract}")
    elif tesseract_available():
        print("  Tesseract    : available on PATH")
    else:
        print("  Tesseract    : unavailable (okay only if Step-03 covers all frames)")

    player = ThermalHoverVideoPlayer(
        video_path,
        config,
        config_source,
        readings_path,
        readings_source,
        args,
    )
    player.run()


if __name__ == "__main__":
    main()
