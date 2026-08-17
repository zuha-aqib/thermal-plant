import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pytesseract


# ============================================================
# VISUAL SETTINGS
# ============================================================

RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LINE_THICKNESS = 2


# ============================================================
# BASIC HELPERS
# ============================================================

def load_json(path):
    """Read a JSON file and return its Python dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def crop_roi(frame, roi):
    """
    Crop an ROI stored as [x, y, width, height].
    """
    x, y, w, h = [int(v) for v in roi]
    return frame[y:y + h, x:x + w]


def parse_temperature(text, decimal_places=1):
    """
    Convert OCR output to a temperature float.

    The thermal-camera display in the example always shows one
    decimal place, e.g.:

        46.7 C
        15.4 C

    OCR occasionally misses the decimal point and returns '467'.
    When decimal_places=1, '467' is therefore interpreted as 46.7.

    Returns:
        float on success
        np.nan on failure
    """
    if text is None:
        return np.nan

    cleaned = text.strip()

    # Keep only digits, minus sign and decimal point.
    cleaned = re.sub(r"[^0-9.\-]", "", cleaned)

    # Find the first plausible numeric token.
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)

    if not match:
        return np.nan

    token = match.group(0)

    try:
        # If OCR saw the decimal point normally, simply parse it.
        if "." in token:
            return float(token)

        # If the display is known to use a fixed number of decimals,
        # restore the missing decimal point.
        if decimal_places > 0:
            sign = -1.0 if token.startswith("-") else 1.0
            digits = token.lstrip("-")

            if not digits:
                return np.nan

            return sign * (int(digits) / (10 ** decimal_places))

        return float(token)

    except ValueError:
        return np.nan


# ============================================================
# OCR PREPROCESSING
# ============================================================

def prepare_temperature_text_crop(crop):
    """
    Prepare a small scale-label crop for Tesseract OCR.

    The sample footage uses GREEN temperature text on a dark
    background. We therefore first try a green-dominance mask.

    If the crop does not contain enough green pixels, the function
    falls back to grayscale thresholding.

    Tesseract generally works better on enlarged black text over a
    clean white background, so that is what this function creates.
    """
    if crop is None or crop.size == 0:
        return None

    # Split OpenCV BGR channels.
    b, g, r = cv2.split(crop)

    # Detect pixels where green is meaningfully stronger than both
    # red and blue. The constants are intentionally modest because
    # video compression can alter the exact RGB values.
    green_mask = (
        (g.astype(np.int16) > r.astype(np.int16) + 10)
        & (g.astype(np.int16) > b.astype(np.int16) + 10)
        & (g > 70)
    )

    green_count = int(np.count_nonzero(green_mask))

    if green_count >= 10:
        # White background, black detected text.
        binary = np.full(crop.shape[:2], 255, dtype=np.uint8)
        binary[green_mask] = 0

        # Light morphological closing can reconnect broken strokes
        # caused by compression without heavily changing the digits.
        kernel = np.ones((2, 2), dtype=np.uint8)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

    else:
        # Fallback for videos where the numeric labels are not green.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # OTSU picks a threshold automatically for the small crop.
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # Tesseract usually prefers dark text on a white background.
        # If this crop is mostly dark, invert it.
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

    # Enlarge the tiny text substantially before OCR.
    binary = cv2.resize(
        binary,
        None,
        fx=4.0,
        fy=4.0,
        interpolation=cv2.INTER_CUBIC,
    )

    # Add a white border so characters close to the crop boundary
    # are less likely to be clipped by Tesseract.
    binary = cv2.copyMakeBorder(
        binary,
        20,
        20,
        20,
        20,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    return binary


def ocr_temperature(frame, roi, decimal_places=1):
    """
    OCR one temperature label from one frame.

    Returns:
        value: float or np.nan
        raw_text: original Tesseract output
        prepared_crop: image used for OCR, useful for debugging
    """
    crop = crop_roi(frame, roi)
    prepared = prepare_temperature_text_crop(crop)

    if prepared is None:
        return np.nan, "", None

    # Restrict OCR to exactly the characters we expect.
    config = (
        "--psm 7 "
        "-c tessedit_char_whitelist=0123456789.-"
    )

    text = pytesseract.image_to_string(
        prepared,
        config=config,
    )

    value = parse_temperature(
        text,
        decimal_places=decimal_places,
    )

    return value, text.strip(), prepared


# ============================================================
# SCALE OCR PASS
# ============================================================

def scan_dynamic_scale(
    video_path,
    scale_config,
    output_dir,
    start_frame=0,
    end_frame=None,
    ocr_every=1,
    min_allowed=-100.0,
    max_allowed=1000.0,
    debug_invalid_limit=20,
):
    """
    FIRST PASS through the video.

    Read the dynamic scale MIN and MAX text values.

    We do this as a separate pass because OCR can occasionally miss
    a frame. After scanning, missing values can be interpolated from
    neighboring valid frames rather than letting one OCR failure
    destroy the temperature calculation for that frame.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if end_frame is None:
        end_frame = total_frames - 1

    start_frame = max(0, int(start_frame))
    end_frame = min(int(end_frame), total_frames - 1)

    if end_frame < start_frame:
        raise ValueError("end_frame must be >= start_frame")

    frame_count = end_frame - start_frame + 1

    raw_min = np.full(frame_count, np.nan, dtype=np.float64)
    raw_max = np.full(frame_count, np.nan, dtype=np.float64)

    raw_min_text = [""] * frame_count
    raw_max_text = [""] * frame_count

    min_roi = scale_config["scale"]["min_text_roi"]
    max_roi = scale_config["scale"]["max_text_roi"]
    decimal_places = int(
        scale_config["scale"].get("display_decimal_places", 1)
    )

    invalid_debug_dir = output_dir / "ocr_debug_invalid"
    invalid_debug_dir.mkdir(parents=True, exist_ok=True)
    invalid_saved = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print("\nPASS 1/2: Reading dynamic scale labels...")
    print(f"Frames: {start_frame} to {end_frame}")
    print(f"OCR every {ocr_every} frame(s)")

    for local_idx in range(frame_count):
        frame_number = start_frame + local_idx

        ok, frame = cap.read()

        if not ok:
            print(f"\nWarning: could not read frame {frame_number}")
            break

        # Skip OCR on unsampled frames when --ocr-every > 1.
        # Those gaps are filled by interpolation after the scan.
        should_ocr = (
            local_idx % max(1, int(ocr_every)) == 0
            or frame_number == end_frame
        )

        if should_ocr:
            min_value, min_text, min_prepared = ocr_temperature(
                frame,
                min_roi,
                decimal_places=decimal_places,
            )

            max_value, max_text, max_prepared = ocr_temperature(
                frame,
                max_roi,
                decimal_places=decimal_places,
            )

            raw_min_text[local_idx] = min_text
            raw_max_text[local_idx] = max_text

            # Basic physical/plausibility validation.
            pair_valid = (
                np.isfinite(min_value)
                and np.isfinite(max_value)
                and min_allowed <= min_value <= max_allowed
                and min_allowed <= max_value <= max_allowed
                and max_value > min_value
            )

            if pair_valid:
                raw_min[local_idx] = min_value
                raw_max[local_idx] = max_value

            else:
                # Save a few failed OCR examples for diagnosis.
                if invalid_saved < debug_invalid_limit:
                    if min_prepared is not None:
                        cv2.imwrite(
                            str(
                                invalid_debug_dir
                                / f"frame_{frame_number:06d}_MIN.png"
                            ),
                            min_prepared,
                        )

                    if max_prepared is not None:
                        cv2.imwrite(
                            str(
                                invalid_debug_dir
                                / f"frame_{frame_number:06d}_MAX.png"
                            ),
                            max_prepared,
                        )

                    invalid_saved += 1

        if local_idx % 100 == 0 or local_idx == frame_count - 1:
            pct = 100.0 * (local_idx + 1) / frame_count
            print(
                f"\rScale OCR: {local_idx + 1}/{frame_count} "
                f"({pct:.1f}%)",
                end="",
                flush=True,
            )

    cap.release()
    print()

    # --------------------------------------------------------
    # Interpolate missing readings.
    # --------------------------------------------------------
    def interpolate_nan(values):
        values = values.astype(np.float64).copy()
        x = np.arange(len(values))
        valid = np.isfinite(values)

        if np.count_nonzero(valid) == 0:
            raise RuntimeError(
                "OCR could not read ANY valid scale values. "
                "Check the MAX/MIN text ROIs and OCR debug crops."
            )

        if np.count_nonzero(valid) == 1:
            # Only one valid reading exists. Use it everywhere so the
            # script can continue, but this should be treated as a
            # diagnostic warning rather than a trustworthy final run.
            values[:] = values[valid][0]
            return values

        values[~valid] = np.interp(
            x[~valid],
            x[valid],
            values[valid],
        )

        return values

    clean_min = interpolate_nan(raw_min)
    clean_max = interpolate_nan(raw_max)

    # Final sanity check after interpolation.
    bad_order = clean_max <= clean_min

    if np.any(bad_order):
        raise RuntimeError(
            "Some interpolated frames have max_temp <= min_temp. "
            "Check scale OCR results before proceeding."
        )

    # --------------------------------------------------------
    # Save the complete OCR audit trail.
    # --------------------------------------------------------
    scale_csv_path = output_dir / "scale_readings.csv"

    with open(scale_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "frame",
            "timestamp_seconds",
            "raw_min_temp",
            "raw_max_temp",
            "clean_min_temp",
            "clean_max_temp",
            "raw_min_text",
            "raw_max_text",
            "ocr_pair_valid",
        ])

        for local_idx in range(frame_count):
            frame_number = start_frame + local_idx

            writer.writerow([
                frame_number,
                frame_number / fps if fps > 0 else "",
                "" if not np.isfinite(raw_min[local_idx]) else raw_min[local_idx],
                "" if not np.isfinite(raw_max[local_idx]) else raw_max[local_idx],
                clean_min[local_idx],
                clean_max[local_idx],
                raw_min_text[local_idx],
                raw_max_text[local_idx],
                int(
                    np.isfinite(raw_min[local_idx])
                    and np.isfinite(raw_max[local_idx])
                ),
            ])

    valid_pairs = int(
        np.count_nonzero(np.isfinite(raw_min) & np.isfinite(raw_max))
    )

    sampled_frames = int(
        math.ceil(frame_count / max(1, int(ocr_every)))
    )

    print(f"Valid OCR pairs: {valid_pairs}/{sampled_frames} sampled frames")
    print(f"Scale audit CSV: {scale_csv_path}")

    return {
        "fps": fps,
        "total_frames": total_frames,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "clean_min": clean_min,
        "clean_max": clean_max,
        "raw_min": raw_min,
        "raw_max": raw_max,
    }


# ============================================================
# COLOR BAR -> INTENSITY -> TEMPERATURE LOOKUP TABLE
# ============================================================

def build_grayscale_temperature_lut(
    frame,
    bar_roi,
    min_temp,
    max_temp,
    inner_fraction=0.20,
):
    """
    Build a 256-entry grayscale-intensity -> temperature lookup.

    IMPORTANT:
    We do NOT assume that intensity is linearly proportional to
    temperature.

    Instead:
        1) Read the ACTUAL color bar from the current frame.
        2) For each vertical row of the bar, measure its grayscale
           intensity.
        3) The top row corresponds to current max_temp.
        4) The bottom row corresponds to current min_temp.
        5) For every possible image intensity 0..255, find the closest
           intensity actually present in the current bar.

    This is ideal for the provided grayscale thermal palette.
    """
    bar = crop_roi(frame, bar_roi)

    if bar is None or bar.size == 0:
        raise RuntimeError("Thermal color bar crop is empty.")

    bar_h, bar_w = bar.shape[:2]

    # Ignore left/right edges of the selected bar because borders or
    # tick marks can contaminate the palette sample.
    left = int(round(bar_w * inner_fraction))
    right = int(round(bar_w * (1.0 - inner_fraction)))

    left = max(0, min(left, bar_w - 1))
    right = max(left + 1, min(right, bar_w))

    inner_bar = bar[:, left:right]

    # Convert the bar crop to grayscale.
    bar_gray = cv2.cvtColor(inner_bar, cv2.COLOR_BGR2GRAY)

    # One representative intensity per vertical row.
    # Median is robust against compression artifacts/noisy pixels.
    palette_intensity = np.median(
        bar_gray,
        axis=1,
    ).astype(np.float32)

    # Temperature corresponding to each row of the color bar.
    # Top = max, bottom = min.
    palette_temperature = np.linspace(
        float(max_temp),
        float(min_temp),
        num=bar_h,
        dtype=np.float32,
    )

    # For every possible grayscale value, find the nearest color-bar
    # row intensity. This makes per-pixel conversion extremely fast.
    levels = np.arange(256, dtype=np.float32)[:, None]

    nearest_row = np.argmin(
        np.abs(levels - palette_intensity[None, :]),
        axis=1,
    )

    lut = palette_temperature[nearest_row]

    return lut


# ============================================================
# ROI MASK + STATISTICS
# ============================================================

def make_polygon_mask(frame_shape, points, erode_pixels=0):
    """
    Convert one saved polygon to a binary mask.

    The mask is 1 inside the equipment region and 0 elsewhere.
    """
    height, width = frame_shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)

    polygon = np.array(points, dtype=np.int32)

    cv2.fillPoly(
        mask,
        [polygon],
        255,
    )

    # Optional erosion lets the user shrink the ROI inward slightly.
    # This can reduce contamination from cool background pixels that
    # lie exactly on the equipment boundary.
    if erode_pixels > 0:
        k = 2 * int(erode_pixels) + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask


def neutral_color_mask(frame, max_channel_difference=18):
    """
    Identify approximately grayscale pixels.

    The thermal image itself is grayscale in the provided footage,
    while UI overlays such as green text and red crosshairs are
    colored.

    We therefore reject pixels whose B/G/R channels differ too much.

    Example:
        thermal gray pixel: (150, 151, 149) -> KEEP
        green UI text:      (20, 220, 30)   -> REJECT
        red crosshair:      (20, 20, 240)   -> REJECT
    """
    frame_i16 = frame.astype(np.int16)

    channel_max = frame_i16.max(axis=2)
    channel_min = frame_i16.min(axis=2)

    spread = channel_max - channel_min

    return spread <= int(max_channel_difference)


def compute_region_statistics(
    temperature_map,
    polygon_mask,
    valid_pixel_mask,
):
    """
    Compute thermal statistics inside one equipment polygon.
    """
    valid = (
        (polygon_mask > 0)
        & valid_pixel_mask
        & np.isfinite(temperature_map)
    )

    values = temperature_map[valid]

    if values.size == 0:
        return None

    return {
        "min_temp": float(np.min(values)),
        "avg_temp": float(np.mean(values)),
        "max_temp": float(np.max(values)),

        # Extra robust statistics are useful later for alerting and
        # debugging, even though only MIN/AVG/MAX are drawn on video.
        "median_temp": float(np.median(values)),
        "p95_temp": float(np.percentile(values, 95)),
        "p99_temp": float(np.percentile(values, 99)),
        "pixel_count": int(values.size),
    }


# ============================================================
# VIDEO OVERLAY HELPERS
# ============================================================

def draw_region_overlay(frame, region, stats):
    """
    Draw the red polygon and MIN/AVG/MAX label for one equipment ROI.
    """
    points = np.array(
        region["pixel_points"],
        dtype=np.int32,
    )

    cv2.polylines(
        frame,
        [points],
        isClosed=True,
        color=RED,
        thickness=LINE_THICKNESS,
    )

    min_x = int(np.min(points[:, 0]))
    min_y = int(np.min(points[:, 1]))

    if stats is None:
        label = f"{region['name']} | NO VALID PIXELS"
    else:
        label = (
            f"{region['name']} | "
            f"Min {stats['min_temp']:.1f} C | "
            f"Avg {stats['avg_temp']:.1f} C | "
            f"Max {stats['max_temp']:.1f} C"
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    # Prefer drawing ABOVE the ROI. If there is not enough room,
    # place the label just inside the top of the ROI.
    text_x = max(0, min_x)
    text_y = min_y - 10

    if text_y - text_h - baseline < 0:
        text_y = min_y + text_h + 10

    # Black background makes the red/white label readable against
    # changing thermal intensities.
    bg_x1 = text_x
    bg_y1 = max(0, text_y - text_h - 6)
    bg_x2 = min(frame.shape[1] - 1, text_x + text_w + 8)
    bg_y2 = min(frame.shape[0] - 1, text_y + baseline + 4)

    cv2.rectangle(
        frame,
        (bg_x1, bg_y1),
        (bg_x2, bg_y2),
        BLACK,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (text_x + 4, text_y),
        font,
        font_scale,
        WHITE,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# SECOND PASS: PIXEL TEMPERATURE EXTRACTION
# ============================================================

def process_video(
    video_path,
    regions_data,
    scale_config,
    scale_scan,
    output_dir,
    erode_pixels=0,
    max_channel_difference=18,
    write_video=True,
):
    """
    SECOND PASS through the video.

    For every frame:
        1) Use cleaned dynamic min/max scale values.
        2) Read the current grayscale color bar.
        3) Build current intensity -> temperature LUT.
        4) Convert all pixels to temperatures.
        5) Compute MIN / AVG / MAX for every saved equipment ROI.
        6) Draw the values above each red polygon.
        7) Save CSV rows and output video.
    """
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(scale_scan["start_frame"])
    end_frame = int(scale_scan["end_frame"])
    frame_count = end_frame - start_frame + 1

    clean_min = scale_scan["clean_min"]
    clean_max = scale_scan["clean_max"]

    bar_roi = scale_config["scale"]["bar_roi"]
    inner_fraction = float(
        scale_config["scale"].get("bar_horizontal_inner_fraction", 0.20)
    )

    regions = regions_data["regions"]

    # --------------------------------------------------------
    # Prebuild polygon masks once because the camera and ROIs are
    # fixed. This is much faster than rebuilding them every frame.
    # --------------------------------------------------------
    region_masks = {}

    for region in regions:
        region_masks[region["id"]] = make_polygon_mask(
            (height, width),
            region["pixel_points"],
            erode_pixels=erode_pixels,
        )

    # --------------------------------------------------------
    # Output video writer.
    # --------------------------------------------------------
    video_writer = None
    output_video_path = output_dir / "thermal_temperature_output.mp4"

    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        video_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            fps,
            (width, height),
        )

        if not video_writer.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not create output video: {output_video_path}"
            )

    # --------------------------------------------------------
    # Per-frame/per-region CSV.
    # --------------------------------------------------------
    log_csv_path = output_dir / "temperature_log.csv"

    log_file = open(
        log_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    )

    writer = csv.writer(log_file)

    writer.writerow([
        "frame",
        "timestamp_seconds",
        "region_id",
        "region_name",
        "scale_min_temp",
        "scale_max_temp",
        "min_temp",
        "avg_temp",
        "max_temp",
        "median_temp",
        "p95_temp",
        "p99_temp",
        "valid_pixel_count",
    ])

    # Store values for an end-of-video summary.
    summary_values = defaultdict(
        lambda: {
            "min": [],
            "avg": [],
            "max": [],
            "p95": [],
            "p99": [],
        }
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print("\nPASS 2/2: Converting thermal pixels to temperatures...")

    for local_idx in range(frame_count):
        frame_number = start_frame + local_idx

        ok, frame = cap.read()

        if not ok:
            print(f"\nWarning: could not read frame {frame_number}")
            break

        min_temp = float(clean_min[local_idx])
        max_temp = float(clean_max[local_idx])

        # ----------------------------------------------------
        # Build current frame's intensity -> temperature mapping.
        # ----------------------------------------------------
        lut = build_grayscale_temperature_lut(
            frame,
            bar_roi=bar_roi,
            min_temp=min_temp,
            max_temp=max_temp,
            inner_fraction=inner_fraction,
        )

        # Convert the whole thermal frame to grayscale intensities.
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # A 256-entry LUT means every pixel conversion is just an
        # array lookup, so this operation is very fast.
        temperature_map = lut[frame_gray]

        # Reject colored UI overlays such as green text/red markers.
        valid_pixel_mask = neutral_color_mask(
            frame,
            max_channel_difference=max_channel_difference,
        )

        output_frame = frame.copy()

        # ----------------------------------------------------
        # Compute stats for every annotated machine/stack.
        # ----------------------------------------------------
        for region in regions:
            polygon_mask = region_masks[region["id"]]

            stats = compute_region_statistics(
                temperature_map,
                polygon_mask,
                valid_pixel_mask,
            )

            if stats is None:
                writer.writerow([
                    frame_number,
                    frame_number / fps if fps > 0 else "",
                    region["id"],
                    region["name"],
                    min_temp,
                    max_temp,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    0,
                ])

            else:
                writer.writerow([
                    frame_number,
                    frame_number / fps if fps > 0 else "",
                    region["id"],
                    region["name"],
                    min_temp,
                    max_temp,
                    stats["min_temp"],
                    stats["avg_temp"],
                    stats["max_temp"],
                    stats["median_temp"],
                    stats["p95_temp"],
                    stats["p99_temp"],
                    stats["pixel_count"],
                ])

                summary_values[region["name"]]["min"].append(
                    stats["min_temp"]
                )
                summary_values[region["name"]]["avg"].append(
                    stats["avg_temp"]
                )
                summary_values[region["name"]]["max"].append(
                    stats["max_temp"]
                )
                summary_values[region["name"]]["p95"].append(
                    stats["p95_temp"]
                )
                summary_values[region["name"]]["p99"].append(
                    stats["p99_temp"]
                )

            if write_video:
                draw_region_overlay(
                    output_frame,
                    region,
                    stats,
                )

        # Also display the dynamic scale values in the upper-left
        # corner so the processed video is easy to audit visually.
        if write_video:
            scale_label = (
                f"Scale: {min_temp:.1f} C to {max_temp:.1f} C"
            )

            cv2.putText(
                output_frame,
                scale_label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                WHITE,
                2,
                cv2.LINE_AA,
            )

            video_writer.write(output_frame)

        if local_idx % 100 == 0 or local_idx == frame_count - 1:
            pct = 100.0 * (local_idx + 1) / frame_count
            print(
                f"\rTemperature extraction: "
                f"{local_idx + 1}/{frame_count} ({pct:.1f}%)",
                end="",
                flush=True,
            )

    cap.release()

    if video_writer is not None:
        video_writer.release()

    log_file.close()
    print()

    # --------------------------------------------------------
    # Save an easy end-of-video summary per equipment region.
    # --------------------------------------------------------
    summary_csv_path = output_dir / "temperature_summary.csv"

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "region_name",
            "lowest_frame_min",
            "mean_of_frame_averages",
            "highest_frame_max",
            "mean_p95",
            "highest_p99",
        ])

        for region_name, values in summary_values.items():
            writer.writerow([
                region_name,
                np.min(values["min"]) if values["min"] else "",
                np.mean(values["avg"]) if values["avg"] else "",
                np.max(values["max"]) if values["max"] else "",
                np.mean(values["p95"]) if values["p95"] else "",
                np.max(values["p99"]) if values["p99"] else "",
            ])

    print("\nFinished.")
    print(f"Temperature log     : {log_csv_path}")
    print(f"Temperature summary : {summary_csv_path}")

    if write_video:
        print(f"Processed video     : {output_video_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-frame MIN/AVG/MAX temperatures from fixed "
            "annotated equipment ROIs in grayscale thermal video."
        )
    )

    parser.add_argument(
        "video",
        help="Path to the ORIGINAL unannotated thermal MP4",
    )

    parser.add_argument(
        "regions_json",
        help="Path to regions.json created by the ROI annotator",
    )

    parser.add_argument(
        "scale_config_json",
        help="Path to scale_config.json created by configure_thermal_scale.py",
    )

    parser.add_argument(
        "--output",
        default="thermal_temperature_output",
        help="Output folder",
    )

    parser.add_argument(
        "--tesseract",
        default=None,
        help=(
            "Optional full path to tesseract executable, e.g. "
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        ),
    )

    parser.add_argument(
        "--ocr-every",
        type=int,
        default=1,
        help=(
            "Run OCR every N frames. Default 1 = every frame. "
            "Skipped frames are linearly interpolated."
        ),
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame to process",
    )

    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame to process. Default = end of video",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Convenience option for testing. Example: --max-frames 300 "
            "processes only 300 frames starting at --start-frame."
        ),
    )

    parser.add_argument(
        "--erode",
        type=int,
        default=0,
        help=(
            "Shrink each polygon inward by this many pixels before "
            "calculating temperature. Default 0."
        ),
    )

    parser.add_argument(
        "--max-channel-difference",
        type=int,
        default=18,
        help=(
            "Reject colored overlay pixels when max(B,G,R)-min(B,G,R) "
            "exceeds this value. Default 18."
        ),
    )

    parser.add_argument(
        "--min-allowed-temp",
        type=float,
        default=-100.0,
        help="Reject OCR values below this temperature",
    )

    parser.add_argument(
        "--max-allowed-temp",
        type=float,
        default=1000.0,
        help="Reject OCR values above this temperature",
    )

    parser.add_argument(
        "--no-video",
        action="store_true",
        help=(
            "Calculate CSV temperatures without writing the processed MP4."
        ),
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    regions_json_path = Path(args.regions_json)
    scale_config_json_path = Path(args.scale_config_json)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If Tesseract is installed but not on PATH, the user can pass
    # --tesseract with the executable location.
    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract

    regions_data = load_json(regions_json_path)
    scale_config = load_json(scale_config_json_path)

    # --------------------------------------------------------
    # Validate that video resolution matches saved configuration.
    # --------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ann_width = int(regions_data["video"]["width"])
    ann_height = int(regions_data["video"]["height"])

    if (width, height) != (ann_width, ann_height):
        raise RuntimeError(
            "Video resolution does not match regions.json.\n"
            f"Video       : {width} x {height}\n"
            f"Annotations : {ann_width} x {ann_height}\n"
            "Use the same ORIGINAL video that was annotated."
        )

    cfg_width = int(scale_config["video"]["width"])
    cfg_height = int(scale_config["video"]["height"])

    if (width, height) != (cfg_width, cfg_height):
        raise RuntimeError(
            "Video resolution does not match scale_config.json.\n"
            f"Video        : {width} x {height}\n"
            f"Scale config : {cfg_width} x {cfg_height}"
        )

    start_frame = max(0, int(args.start_frame))
    end_frame = args.end_frame

    if end_frame is None:
        end_frame = total_frames - 1

    if args.max_frames is not None:
        end_frame = min(
            int(end_frame),
            start_frame + int(args.max_frames) - 1,
        )

    # --------------------------------------------------------
    # PASS 1: dynamic scale OCR.
    # --------------------------------------------------------
    scale_scan = scan_dynamic_scale(
        video_path=video_path,
        scale_config=scale_config,
        output_dir=output_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        ocr_every=max(1, int(args.ocr_every)),
        min_allowed=float(args.min_allowed_temp),
        max_allowed=float(args.max_allowed_temp),
    )

    # --------------------------------------------------------
    # PASS 2: per-pixel thermal conversion + ROI statistics.
    # --------------------------------------------------------
    process_video(
        video_path=video_path,
        regions_data=regions_data,
        scale_config=scale_config,
        scale_scan=scale_scan,
        output_dir=output_dir,
        erode_pixels=max(0, int(args.erode)),
        max_channel_difference=max(
            0,
            int(args.max_channel_difference),
        ),
        write_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
