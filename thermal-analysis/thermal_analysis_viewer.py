#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Reuse the exact Step-03 thermal/OCR math already validated in this repo.
# This file lives in thermal-analysis/, while Step 03 lives one folder above.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import step_03_process_thermal_temperatures as t3

WINDOW_NAME = "Thermal Analysis - Hover + ROI Stats + Hotspots"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)

MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 820


# =============================================================================
# FILE / PATH HELPERS
# =============================================================================

def load_json(path: Path) -> dict:
    return t3.load_json(path)


def find_raw_videos_ancestor(video_path: Path) -> Optional[Path]:
    video_path = video_path.resolve()
    for parent in video_path.parents:
        if parent.name.lower() == "raw-videos":
            return parent
    return None


def discover_scale_readings(video_path: Path, explicit: Optional[str]):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, "EXPLICIT STEP-03 CSV"

    raw_root = find_raw_videos_ancestor(video_path)
    if raw_root is None:
        return None, "NONE"

    project_root = raw_root.parent
    relative_video = video_path.resolve().relative_to(raw_root)
    relative_parent = relative_video.parent

    for root_name in (
        "step-03-roi-videos",
        "step03-roi-videos",
        "step-3-roi-videos",
    ):
        candidate = (
            project_root
            / root_name
            / relative_parent
            / video_path.stem
            / "scale_readings.csv"
        )
        if candidate.is_file():
            return candidate.resolve(), "STEP 03 CLEAN SCALE"

    return None, "NONE"


def load_scale_readings(path: Path, total_frames: int):
    mins = np.full(total_frames, np.nan, dtype=np.float32)
    maxs = np.full(total_frames, np.nan, dtype=np.float32)
    loaded = 0

    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"frame", "clean_min_temp_used", "clean_max_temp_used"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(
                "scale_readings.csv is missing frame / clean_min_temp_used / "
                "clean_max_temp_used columns."
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
            if not (
                np.isfinite(min_temp)
                and np.isfinite(max_temp)
                and max_temp > min_temp
            ):
                continue

            mins[frame_number] = min_temp
            maxs[frame_number] = max_temp
            loaded += 1

    return mins, maxs, loaded


# =============================================================================
# BACKGROUND SCALE PROVIDER
# =============================================================================

class ScaleProvider:
    """Prefer Step-03 cleaned scale; otherwise perform Step-03-style live OCR."""

    def __init__(
        self,
        scale_config: dict,
        fps: float,
        total_frames: int,
        readings_path: Optional[Path],
        args,
    ):
        self.scale = scale_config["scale"]
        self.args = args
        self.fps = float(fps)
        self.total_frames = int(total_frames)

        self.cached_min = np.full(total_frames, np.nan, dtype=np.float32)
        self.cached_max = np.full(total_frames, np.nan, dtype=np.float32)
        self.cached_rows = 0

        if readings_path is not None:
            self.cached_min, self.cached_max, self.cached_rows = load_scale_readings(
                readings_path, total_frames
            )

        self.interval_frames = max(1, int(round(self.fps / float(args.ocr_hz))))
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="thermal-ocr")
        self.future = None
        self.future_frame = None
        self.last_submit_frame = -(10**9)

        self.trusted_min = self._finite_or_none(self.scale.get("reference_ocr_min_temp"))
        self.trusted_max = self._finite_or_none(self.scale.get("reference_ocr_max_temp"))
        self.status = "REFERENCE SCALE" if self.trusted_min is not None else "WAITING FOR OCR"

    @staticmethod
    def _finite_or_none(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)

    def cached(self, frame_number: int):
        if not 0 <= frame_number < self.total_frames:
            return None
        mn = float(self.cached_min[frame_number])
        mx = float(self.cached_max[frame_number])
        if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
            return mn, mx
        return None

    def _ocr_worker(self, frame: np.ndarray):
        decimal_places = int(self.scale.get("display_decimal_places", 1))
        padding = int(self.scale.get("ocr_padding_pixels", 6))

        min_method = self.scale.get("min_ocr_method", "green_difference")
        max_method = self.scale.get("max_ocr_method", "green_difference")
        min_psm = int(self.scale.get("min_ocr_psm", 7))
        max_psm = int(self.scale.get("max_ocr_psm", 7))

        min_value, min_text, _ = t3.ocr_temperature(
            frame,
            self.scale["min_text_roi"],
            decimal_places=decimal_places,
            method=min_method,
            psm=min_psm,
            padding=padding,
        )
        max_value, max_text, _ = t3.ocr_temperature(
            frame,
            self.scale["max_text_roi"],
            decimal_places=decimal_places,
            method=max_method,
            psm=max_psm,
            padding=padding,
        )

        if t3._basic_scale_pair_valid(
            min_value,
            max_value,
            self.args.min_allowed_temp,
            self.args.max_allowed_temp,
            self.args.min_scale_span,
        ):
            return {
                "min": float(min_value),
                "max": float(max_value),
                "source": "PRIMARY OCR",
            }

        min_candidates = t3._collect_ocr_candidates(
            frame=frame,
            roi=self.scale["min_text_roi"],
            decimal_places=decimal_places,
            primary_method=min_method,
            primary_psm=min_psm,
            padding=padding,
        )
        max_candidates = t3._collect_ocr_candidates(
            frame=frame,
            roi=self.scale["max_text_roi"],
            decimal_places=decimal_places,
            primary_method=max_method,
            primary_psm=max_psm,
            padding=padding,
        )

        best = t3._choose_temporally_plausible_pair(
            min_candidates=min_candidates,
            max_candidates=max_candidates,
            previous_min=None,
            previous_max=None,
            min_allowed=self.args.min_allowed_temp,
            max_allowed=self.args.max_allowed_temp,
            min_scale_span=self.args.min_scale_span,
            max_jump=1e9,
        )

        if best is None:
            return None

        best_min, best_max = best
        return {
            "min": float(best_min["value"]),
            "max": float(best_max["value"]),
            "source": "RETRY OCR",
        }

    def _accept(self, result, allow_jump=False):
        if result is None:
            self.status = "OCR FAILED - HOLD PREVIOUS"
            return

        mn = float(result["min"])
        mx = float(result["max"])

        if (
            not allow_jump
            and self.trusted_min is not None
            and self.trusted_max is not None
            and float(self.args.max_scale_jump) > 0
        ):
            if (
                abs(mn - self.trusted_min) > float(self.args.max_scale_jump)
                or abs(mx - self.trusted_max) > float(self.args.max_scale_jump)
            ):
                self.status = "OCR JUMP REJECTED - HOLD PREVIOUS"
                return

        self.trusted_min = mn
        self.trusted_max = mx
        self.status = result.get("source", "LIVE OCR")

    def poll(self):
        if self.future is None or not self.future.done():
            return

        future = self.future
        self.future = None
        self.future_frame = None

        try:
            result = future.result()
        except Exception:
            self.status = "OCR ERROR - HOLD PREVIOUS"
            return

        self._accept(result, allow_jump=False)

    def get(self, frame_number: int, frame: np.ndarray, force=False):
        cached = self.cached(frame_number)
        if cached is not None:
            return cached[0], cached[1], "STEP 03 CLEAN SCALE"

        self.poll()

        if force and (self.trusted_min is None or self.trusted_max is None):
            self._accept(self._ocr_worker(frame.copy()), allow_jump=True)

        if (
            self.future is None
            and frame_number - self.last_submit_frame >= self.interval_frames
        ):
            self.last_submit_frame = int(frame_number)
            self.future_frame = int(frame_number)
            self.future = self.executor.submit(self._ocr_worker, frame.copy())

        if self.trusted_min is None or self.trusted_max is None:
            return None, None, self.status

        return float(self.trusted_min), float(self.trusted_max), self.status

    def reseed(self, frame_number: int, frame: np.ndarray):
        cached = self.cached(frame_number)
        if cached is not None:
            return
        try:
            self._accept(self._ocr_worker(frame.copy()), allow_jump=True)
            self.last_submit_frame = int(frame_number)
        except Exception:
            pass

    def reset_reference(self):
        self.trusted_min = self._finite_or_none(self.scale.get("reference_ocr_min_temp"))
        self.trusted_max = self._finite_or_none(self.scale.get("reference_ocr_max_temp"))
        self.status = "REFERENCE SCALE" if self.trusted_min is not None else "WAITING FOR OCR"
        self.last_submit_frame = -(10**9)


# =============================================================================
# ROI ANALYSIS
# =============================================================================

class RegionAnalyzer:
    def __init__(self, regions, width, height, erode_pixels=0):
        self.regions = regions
        self.width = int(width)
        self.flat_indices = {}

        for region in regions:
            mask = t3.make_polygon_mask(
                (height, width),
                region["pixel_points"],
                erode_pixels=erode_pixels,
            )
            self.flat_indices[int(region["id"])] = np.flatnonzero(mask.reshape(-1) > 0)

    def analyze(self, temperature_map, valid_pixel_mask):
        temp_flat = temperature_map.reshape(-1)
        valid_flat = valid_pixel_mask.reshape(-1)
        output = {}

        for region in self.regions:
            region_id = int(region["id"])
            indices = self.flat_indices[region_id]

            if indices.size == 0:
                output[region_id] = None
                continue

            valid = valid_flat[indices]
            if not np.any(valid):
                output[region_id] = None
                continue

            valid_indices = indices[valid]
            values = temp_flat[valid_indices]
            finite = np.isfinite(values)

            if not np.any(finite):
                output[region_id] = None
                continue

            valid_indices = valid_indices[finite]
            values = values[finite]
            local_max = int(np.argmax(values))
            max_flat = int(valid_indices[local_max])
            max_y, max_x = divmod(max_flat, self.width)

            output[region_id] = {
                "min_temp": float(np.min(values)),
                "avg_temp": float(np.mean(values)),
                "max_temp": float(values[local_max]),
                "max_location": (int(max_x), int(max_y)),
                "pixel_count": int(values.size),
            }

        return output


def sample_hover_3x3(frame, temperature_map, valid_mask, x, y, scale):
    height, width = frame.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return None, "OUTSIDE FRAME", 0

    for roi in (scale["bar_roi"], scale["max_text_roi"], scale["min_text_roi"]):
        rx, ry, rw, rh = [int(v) for v in roi]
        if rx <= x < rx + rw and ry <= y < ry + rh:
            return None, "THERMAL SCALE / UI", 0

    x1, x2 = max(0, x - 1), min(width, x + 2)
    y1, y2 = max(0, y - 1), min(height, y + 2)

    temps = temperature_map[y1:y2, x1:x2]
    valid = valid_mask[y1:y2, x1:x2] & np.isfinite(temps)
    values = temps[valid]

    if values.size == 0:
        return None, "NO VALID THERMAL PIXELS", 0

    return float(np.mean(values)), "OK", int(values.size)


# =============================================================================
# DRAWING
# =============================================================================

def draw_text_box(image, lines, x, y, border=WHITE, font_scale=0.45):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    line_h = 19
    widths = [cv2.getTextSize(str(line), font, font_scale, thickness)[0][0] for line in lines]
    box_w = max(widths, default=100) + 16
    box_h = line_h * len(lines) + 10

    x = max(0, min(int(x), image.shape[1] - box_w - 1))
    y = max(box_h, min(int(y), image.shape[0] - 1))
    y1 = max(0, y - box_h)
    x2 = min(image.shape[1] - 1, x + box_w)

    overlay = image.copy()
    cv2.rectangle(overlay, (x, y1), (x2, y), BLACK, -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
    cv2.rectangle(image, (x, y1), (x2, y), border, 1)

    text_y = y1 + 17
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
        text_y += line_h


def draw_region(image, region, stats, sx, sy):
    points = np.asarray(region["pixel_points"], dtype=np.float32)
    display_points = points.copy()
    display_points[:, 0] *= sx
    display_points[:, 1] *= sy
    display_points = np.round(display_points).astype(np.int32)

    cv2.polylines(image, [display_points], True, RED, 2)

    min_x = int(np.min(display_points[:, 0]))
    min_y = int(np.min(display_points[:, 1]))

    if stats is None:
        lines = [region["name"], "NO VALID THERMAL PIXELS"]
    else:
        lines = [
            region["name"],
            (
                f"Min {stats['min_temp']:.1f} C | "
                f"Avg {stats['avg_temp']:.1f} C | "
                f"Max {stats['max_temp']:.1f} C"
            ),
        ]

    draw_text_box(image, lines, min_x, max(44, min_y - 5), border=RED, font_scale=0.42)

    if stats is None:
        return

    max_x, max_y = stats["max_location"]
    dx = int(round(max_x * sx))
    dy = int(round(max_y * sy))

    cv2.circle(image, (dx, dy), 6, RED, -1, cv2.LINE_AA)
    cv2.circle(image, (dx, dy), 9, WHITE, 1, cv2.LINE_AA)

    label = f"MAX {stats['max_temp']:.1f} C"
    tx = min(image.shape[1] - 120, dx + 12)
    ty = max(20, dy - 8)
    cv2.putText(image, label, (max(0, tx), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.46, BLACK, 3, cv2.LINE_AA)
    cv2.putText(image, label, (max(0, tx), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.46, RED, 1, cv2.LINE_AA)


def draw_hover(image, x, y, lines, valid=True):
    color = GREEN if valid else RED
    cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 20, 2)
    draw_text_box(image, lines, x + 14, y + 80, border=color, font_scale=0.45)


def draw_controls(image):
    text = "SPACE Play/Pause | A/D frame | J/L 1 sec | R Restart | S Screenshot | Q/ESC Quit"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.41
    (w, h), baseline = cv2.getTextSize(text, font, scale, 1)
    y1 = max(0, image.shape[0] - h - baseline - 12)
    cv2.rectangle(image, (0, y1), (min(image.shape[1] - 1, w + 18), image.shape[0] - 1), BLACK, -1)
    cv2.putText(image, text, (8, image.shape[0] - 7), font, scale, WHITE, 1, cv2.LINE_AA)


# =============================================================================
# PLAYER
# =============================================================================

class ThermalAnalysisPlayer:
    def __init__(self, video_path, regions_data, scale_config, readings_path, readings_source, args):
        self.video_path = Path(video_path).resolve()
        self.regions_data = regions_data
        self.regions = list(regions_data["regions"])
        self.scale_config = scale_config
        self.scale = scale_config["scale"]
        self.readings_path = readings_path
        self.readings_source = readings_source
        self.args = args

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.fps <= 0 or self.total_frames <= 0:
            raise RuntimeError("Video reported invalid FPS/frame count.")

        self._validate_inputs()

        self.scale_provider = ScaleProvider(
            scale_config,
            self.fps,
            self.total_frames,
            readings_path,
            args,
        )
        self.region_analyzer = RegionAnalyzer(
            self.regions,
            self.width,
            self.height,
            erode_pixels=max(0, int(args.erode)),
        )

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Could not read first frame.")

        self.frame_index = 0
        self.current_frame = frame
        self.playing = not args.start_paused
        self.loop_count = 0
        self.next_deadline = time.perf_counter()

        self.current_min = None
        self.current_max = None
        self.scale_source = ""
        self.temperature_map = None
        self.valid_mask = None
        self.region_stats = {}

        self.mouse_x = None
        self.mouse_y = None
        self.display_width = None
        self.display_height = None
        self.pending_seek = None
        self.updating_trackbar = False

        self.screenshot_dir = Path(args.screenshot_dir).resolve()
        self._update_analysis(force_scale=True)

    def _validate_inputs(self):
        ann = self.regions_data.get("video", {})
        if (int(ann.get("width", self.width)), int(ann.get("height", self.height))) != (self.width, self.height):
            raise RuntimeError("regions.json resolution does not match this video.")

        cfg = self.scale_config.get("video", {})
        if (int(cfg.get("width", self.width)), int(cfg.get("height", self.height))) != (self.width, self.height):
            raise RuntimeError("scale_config.json resolution does not match this video.")

        if not self.regions:
            raise RuntimeError("regions.json contains no regions.")

        for field in ("bar_roi", "max_text_roi", "min_text_roi"):
            if field not in self.scale:
                raise RuntimeError(f"scale_config.json is missing: {field}")

    def close(self):
        self.scale_provider.close()
        self.cap.release()
        cv2.destroyAllWindows()

    def _update_analysis(self, force_scale=False):
        mn, mx, source = self.scale_provider.get(
            self.frame_index,
            self.current_frame,
            force=force_scale,
        )
        self.current_min = mn
        self.current_max = mx
        self.scale_source = source
        self.temperature_map = None
        self.valid_mask = None
        self.region_stats = {}

        if mn is None or mx is None:
            return

        lut = t3.build_grayscale_temperature_lut(
            self.current_frame,
            bar_roi=self.scale["bar_roi"],
            min_temp=mn,
            max_temp=mx,
            inner_fraction=float(self.scale.get("bar_horizontal_inner_fraction", 0.20)),
        )
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        self.temperature_map = lut[gray]
        self.valid_mask = t3.neutral_color_mask(
            self.current_frame,
            max_channel_difference=int(self.args.max_channel_difference),
        )
        self.region_stats = self.region_analyzer.analyze(self.temperature_map, self.valid_mask)

    def _load_frame(self, frame_number, pause=True, reseed=True):
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

        if reseed and self.scale_provider.cached(frame_number) is None:
            self.scale_provider.reseed(frame_number, frame)

        self._update_analysis(force_scale=False)

    def _restart_loop(self):
        self.loop_count += 1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.cap.read()
        if not ok:
            self.playing = False
            return
        self.frame_index = 0
        self.current_frame = frame
        self.scale_provider.reset_reference()
        self._update_analysis(force_scale=False)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        self.next_deadline = time.perf_counter() + 1.0 / self.fps

    def _read_next(self):
        if self.frame_index >= self.total_frames - 1:
            if self.args.loop:
                self._restart_loop()
            else:
                self.playing = False
            return

        ok, frame = self.cap.read()
        if not ok:
            if self.args.loop:
                self._restart_loop()
            else:
                self.playing = False
            return

        self.frame_index += 1
        self.current_frame = frame
        self._update_analysis(force_scale=False)

    def _toggle_play(self):
        if self.playing:
            self.playing = False
            return

        if self.frame_index >= self.total_frames - 1:
            if self.args.loop:
                self._restart_loop()
            else:
                self._load_frame(0, pause=False, reseed=True)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index + 1)
        self.playing = True
        self.next_deadline = time.perf_counter()

    def _advance_if_due(self):
        if not self.playing:
            self.scale_provider.poll()
            return

        now = time.perf_counter()
        if now < self.next_deadline:
            self.scale_provider.poll()
            return

        self._read_next()
        period = 1.0 / self.fps
        self.next_deadline += period
        if now - self.next_deadline > 0.5:
            self.next_deadline = now + period

    def _display_size(self):
        scale = min(MAX_DISPLAY_WIDTH / self.width, MAX_DISPLAY_HEIGHT / self.height, 1.0)
        return max(1, int(round(self.width * scale))), max(1, int(round(self.height * scale)))

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = int(x)
            self.mouse_y = int(y)

    def _mouse_original(self):
        if self.mouse_x is None or self.display_width is None:
            return None
        if not (0 <= self.mouse_x < self.display_width and 0 <= self.mouse_y < self.display_height):
            return None
        x = int(round(self.mouse_x * self.width / self.display_width))
        y = int(round(self.mouse_y * self.height / self.display_height))
        return max(0, min(x, self.width - 1)), max(0, min(y, self.height - 1))

    def _render(self):
        self.display_width, self.display_height = self._display_size()
        display = cv2.resize(
            self.current_frame,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA if self.display_width < self.width else cv2.INTER_LINEAR,
        )
        sx = self.display_width / self.width
        sy = self.display_height / self.height

        if self.temperature_map is not None:
            for region in self.regions:
                draw_region(
                    display,
                    region,
                    self.region_stats.get(int(region["id"])),
                    sx,
                    sy,
                )

        state = "PLAYING" if self.playing else "PAUSED"
        timestamp = self.frame_index / self.fps
        duration = self.total_frames / self.fps
        status = [
            f"{state} | HOVER + ROI STATS + HOTSPOTS",
            f"Frame {self.frame_index:,}/{self.total_frames - 1:,} | {timestamp:.1f}s / {duration:.1f}s | loop {self.loop_count}",
        ]
        if self.current_min is not None:
            status.append(f"Scale {self.current_min:.1f}..{self.current_max:.1f} C | {self.scale_source}")
        else:
            status.append(f"Scale unavailable | {self.scale_source}")
        draw_text_box(display, status, 10, 90, border=YELLOW, font_scale=0.44)

        mouse = self._mouse_original()
        if mouse is not None:
            ox, oy = mouse
            if self.temperature_map is None or self.valid_mask is None:
                draw_hover(
                    display,
                    self.mouse_x,
                    self.mouse_y,
                    [f"Pixel x={ox}, y={oy}", "Temperature unavailable"],
                    valid=False,
                )
            else:
                value, reason, count = sample_hover_3x3(
                    self.current_frame,
                    self.temperature_map,
                    self.valid_mask,
                    ox,
                    oy,
                    self.scale,
                )
                if value is not None:
                    draw_hover(
                        display,
                        self.mouse_x,
                        self.mouse_y,
                        [f"3x3 avg: {value:.1f} C", f"Pixel x={ox}, y={oy}", f"Valid: {count}/9"],
                        valid=True,
                    )
                else:
                    draw_hover(
                        display,
                        self.mouse_x,
                        self.mouse_y,
                        [f"Pixel x={ox}, y={oy}", reason, "Temperature: N/A"],
                        valid=False,
                    )

        draw_controls(display)
        return display

    def _trackbar_callback(self, position):
        if not self.updating_trackbar:
            self.pending_seek = int(position)

    def _update_trackbar(self):
        self.updating_trackbar = True
        cv2.setTrackbarPos("Frame", WINDOW_NAME, int(self.frame_index))
        self.updating_trackbar = False

    def save_screenshot(self, display):
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.screenshot_dir / f"{self.video_path.stem}_frame_{self.frame_index:06d}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        if cv2.imwrite(str(path), display):
            print(f"\nScreenshot saved: {path}")

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Frame", WINDOW_NAME, 0, max(1, self.total_frames - 1), self._trackbar_callback)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        print("\n============================================================")
        print("THERMAL ANALYSIS VIEWER")
        print("============================================================")
        print(f"Video          : {self.video_path}")
        print(f"Resolution     : {self.width} x {self.height}")
        print(f"FPS            : {self.fps:.3f}")
        print(f"Regions        : {len(self.regions)}")
        print(f"Step-03 scale  : {self.readings_source}")
        print(f"Cached scales  : {self.scale_provider.cached_rows}/{self.total_frames}")
        print(f"Auto-loop      : {'ON' if self.args.loop else 'OFF'}")
        print("\nActive per frame:")
        print("  - 3x3 mouse hover")
        print("  - ROI MIN / AVG / MAX")
        print("  - Red max-temperature hotspot")

        try:
            while True:
                if self.pending_seek is not None:
                    requested = self.pending_seek
                    self.pending_seek = None
                    if requested != self.frame_index:
                        self._load_frame(requested, pause=True, reseed=True)

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
                elif key in (ord("a"), ord("A")):
                    self._load_frame(self.frame_index - 1, pause=True, reseed=True)
                elif key in (ord("d"), ord("D")):
                    self._load_frame(self.frame_index + 1, pause=True, reseed=True)
                elif key in (ord("j"), ord("J")):
                    self._load_frame(self.frame_index - int(round(self.fps)), pause=True, reseed=True)
                elif key in (ord("l"), ord("L")):
                    self._load_frame(self.frame_index + int(round(self.fps)), pause=True, reseed=True)
                elif key in (ord("r"), ord("R")):
                    self._load_frame(0, pause=not self.playing, reseed=True)
                elif key in (ord("s"), ord("S")):
                    self.save_screenshot(display)
        finally:
            self.close()


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive thermal analysis: continuous hover + ROI MIN/AVG/MAX + "
            "red maximum-temperature hotspot."
        )
    )
    parser.add_argument("video", help="Original rendered thermal MP4")
    parser.add_argument("regions_json", help="Step-01 regions.json")
    parser.add_argument("scale_config_json", help="Step-02 scale_config.json")
    parser.add_argument("--scale-readings", default=None, help="Optional Step-03 scale_readings.csv")
    parser.add_argument("--tesseract", default=None, help="Optional full path to tesseract.exe")
    parser.add_argument("--ocr-hz", type=float, default=3.0)
    parser.add_argument("--max-scale-jump", type=float, default=2.5)
    parser.add_argument("--min-allowed-temp", type=float, default=-100.0)
    parser.add_argument("--max-allowed-temp", type=float, default=1000.0)
    parser.add_argument("--min-scale-span", type=float, default=5.0)
    parser.add_argument("--max-channel-difference", type=int, default=18)
    parser.add_argument("--erode", type=int, default=0)
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--no-loop", dest="loop", action="store_false")
    parser.add_argument("--screenshot-dir", default="thermal-analysis/screenshots")
    parser.set_defaults(loop=True)
    return parser


def main():
    args = build_parser().parse_args()

    video_path = Path(args.video).expanduser().resolve()
    regions_path = Path(args.regions_json).expanduser().resolve()
    scale_path = Path(args.scale_config_json).expanduser().resolve()

    for path in (video_path, regions_path, scale_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.tesseract:
        t3.pytesseract.pytesseract.tesseract_cmd = args.tesseract

    regions_data = load_json(regions_path)
    scale_config = load_json(scale_path)

    readings_path, readings_source = discover_scale_readings(
        video_path,
        args.scale_readings,
    )

    print("\nInputs:")
    print(f"  Video        : {video_path}")
    print(f"  Regions      : {regions_path}")
    print(f"  Scale config : {scale_path}")
    print(f"  Step-03 CSV  : {readings_source}" + (f" -> {readings_path}" if readings_path else ""))

    player = ThermalAnalysisPlayer(
        video_path,
        regions_data,
        scale_config,
        readings_path,
        readings_source,
        args,
    )
    player.run()


if __name__ == "__main__":
    main()
