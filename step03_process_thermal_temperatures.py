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

# Default Stage-3 output root.
DEFAULT_OUTPUT_ROOT = "step03-roi-videos"


# ============================================================
# OUTPUT PATH HELPERS
# ============================================================

def find_raw_videos_ancestor(video_path):
    """
    Find the nearest parent directory literally named 'raw-videos'.

    This lets every stage preserve nested folders from raw-videos.

    Example:
        raw-videos/
            Furnace/
                Camera-01/
                    sample.mp4

    becomes:
        <step-output-root>/
            Furnace/
                Camera-01/
                    sample/
    """
    video_path = Path(video_path).resolve()

    for parent in video_path.parents:
        if parent.name.lower() == "raw-videos":
            return parent

    return None


def build_video_output_dir(video_path, output_root):
    """
    Build the final per-video output folder.

    If the video is under raw-videos/, preserve all nested folders
    below raw-videos/. Otherwise use only the video's stem.
    """
    video_path = Path(video_path).resolve()
    output_root = Path(output_root).resolve()

    raw_root = find_raw_videos_ancestor(video_path)

    if raw_root is not None:
        relative_video = video_path.relative_to(raw_root)
        relative_parent = relative_video.parent
    else:
        relative_parent = Path()

    return output_root / relative_parent / video_path.stem


# ============================================================
# VISUAL SETTINGS
# ============================================================

RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LINE_THICKNESS = 2


# ============================================================
# OCR RETRY SETTINGS
# ============================================================

# The normal frame uses the OCR method selected during configuration.
#
# ONLY when that reading is missing or implausible do we retry the
# same frame with these alternative preprocessing/PSM combinations.
# This keeps the full-video run reasonably fast while still giving
# difficult frames several chances to be read correctly.
OCR_RETRY_CONFIGS = [
    ("green_difference", 7),
    ("green_difference", 8),
    ("hsv_green", 7),
    ("green_channel_otsu", 7),
    ("gray_otsu", 7),
    ("gray_adaptive", 7),
]


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

    IMPORTANT FIX:
    The old version treated a one-digit OCR result such as "1" as
    0.1 C and "8" as 0.8 C. That is dangerous because a partially
    read value like 15.6 C can easily be reduced to a single digit
    by OCR.

    New rule:
      - If a decimal point is present, parse normally.
      - If the decimal point is missing, only restore it when the
        OCR result contains enough digits to plausibly represent
        the complete display value.

    For one decimal place:
        "156" -> 15.6
        "469" -> 46.9

    But:
        "1"   -> INVALID
        "8"   -> INVALID
        "15"  -> INVALID

    Returning NaN is much safer than silently inventing 0.1 C.
    """
    if text is None:
        return np.nan

    cleaned = text.strip()
    cleaned = re.sub(r"[^0-9.\-]", "", cleaned)

    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)

    if not match:
        return np.nan

    token = match.group(0)

    try:
        if "." in token:
            return float(token)

        if decimal_places > 0:
            sign = -1.0 if token.startswith("-") else 1.0
            digits = token.lstrip("-")

            # Require at least:
            #   two whole-number digits + the requested decimals.
            #
            # With one decimal place that means at least 3 digits:
            #   156 -> 15.6
            #   469 -> 46.9
            minimum_digits = decimal_places + 2

            if len(digits) < minimum_digits:
                return np.nan

            return sign * (
                int(digits) / (10 ** decimal_places)
            )

        return float(token)

    except ValueError:
        return np.nan


# ============================================================
# OCR PREPROCESSING
# ============================================================

def crop_roi_with_padding(frame, roi, padding=0):
    """
    Crop [x, y, width, height] while expanding it slightly.

    A few pixels of padding are useful because a tight manual crop
    can accidentally clip the first digit, which is exactly the kind
    of failure that can turn 15.6 into "6" or 46.9 into "9".
    """
    x, y, w, h = [int(v) for v in roi]
    padding = max(0, int(padding))

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame.shape[1], x + w + padding)
    y2 = min(frame.shape[0], y + h + padding)

    return frame[y1:y2, x1:x2]


def _finish_binary_for_tesseract(binary):
    """
    Normalize any binary text mask to:
        BLACK text on WHITE background,
    enlarge it, and add a white border.
    """
    if binary is None or binary.size == 0:
        return None

    # If the image is mostly dark, it probably contains white text
    # on a black background. Invert it for Tesseract.
    if float(np.mean(binary)) < 127.0:
        binary = cv2.bitwise_not(binary)

    binary = cv2.resize(
        binary,
        None,
        fx=5.0,
        fy=5.0,
        interpolation=cv2.INTER_CUBIC,
    )

    binary = cv2.copyMakeBorder(
        binary,
        24,
        24,
        24,
        24,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    return binary


def prepare_temperature_text_crop(crop, method="green_difference"):
    """
    Prepare a scale-label crop for Tesseract.

    The thermal camera uses pale green/mint text over a dark
    background. Different videos/compression levels can change the
    exact pixel values, so the OCR configuration script tests several
    methods and stores the best method in scale_config.json.

    Supported methods:
        green_difference
        hsv_green
        green_channel_otsu
        gray_otsu
        gray_adaptive
    """
    if crop is None or crop.size == 0:
        return None

    b, g, r = cv2.split(crop)

    if method == "green_difference":
        # Detect pixels where green is even slightly stronger than
        # red/blue. The old threshold was too strict for pale mint
        # anti-aliased text.
        b16 = b.astype(np.int16)
        g16 = g.astype(np.int16)
        r16 = r.astype(np.int16)

        green_advantage = g16 - np.maximum(r16, b16)

        mask = (
            (green_advantage >= 4)
            & (g >= 70)
        ).astype(np.uint8) * 255

        # IMPORTANT:
        # Morphology is done while the TEXT is white foreground.
        # The previous script performed closing on black text over
        # white background, which can erase thin character strokes.
        kernel = np.ones((2, 2), dtype=np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        mask = cv2.dilate(
            mask,
            kernel,
            iterations=1,
        )

        return _finish_binary_for_tesseract(mask)

    if method == "hsv_green":
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Fairly broad range because the label is pale green rather
        # than strongly saturated green.
        lower = np.array([25, 12, 65], dtype=np.uint8)
        upper = np.array([100, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(
            hsv,
            lower,
            upper,
        )

        kernel = np.ones((2, 2), dtype=np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        return _finish_binary_for_tesseract(mask)

    if method == "green_channel_otsu":
        # The green channel usually gives the best contrast for the
        # mint labels even when saturation is low.
        channel = cv2.GaussianBlur(
            g,
            (3, 3),
            0,
        )

        _, binary = cv2.threshold(
            channel,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        return _finish_binary_for_tesseract(binary)

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    if method == "gray_otsu":
        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        return _finish_binary_for_tesseract(binary)

    if method == "gray_adaptive":
        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            5,
        )

        return _finish_binary_for_tesseract(binary)

    raise ValueError(
        f"Unknown OCR preprocessing method: {method}"
    )


def ocr_temperature(
    frame,
    roi,
    decimal_places=1,
    method="green_difference",
    psm=7,
    padding=6,
):
    """
    OCR one temperature label from one frame.

    Returns:
        value
        raw_text
        prepared_crop
    """
    crop = crop_roi_with_padding(
        frame,
        roi,
        padding=padding,
    )

    prepared = prepare_temperature_text_crop(
        crop,
        method=method,
    )

    if prepared is None:
        return np.nan, "", None

    config = (
        f"--psm {int(psm)} "
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

def _basic_scale_pair_valid(
    min_value,
    max_value,
    min_allowed,
    max_allowed,
    min_scale_span,
):
    """
    Basic physical/plausibility validation that does NOT use
    temporal history.
    """
    return (
        np.isfinite(min_value)
        and np.isfinite(max_value)
        and min_allowed <= min_value <= max_allowed
        and min_allowed <= max_value <= max_allowed
        and max_value > min_value
        and (max_value - min_value) >= float(min_scale_span)
    )


def _temporal_scale_pair_valid(
    min_value,
    max_value,
    previous_min,
    previous_max,
    max_jump,
):
    """
    Check whether the new OCR reading is close enough to the most
    recently trusted scale.

    For this thermal camera, the displayed scale changes gradually.
    A jump such as:

        46.8 -> 467.7

    is therefore treated as OCR corruption, not as a real thermal
    scale change.

    max_jump applies independently to MIN and MAX.
    """
    if previous_min is None or previous_max is None:
        return True

    return (
        abs(float(min_value) - float(previous_min)) <= float(max_jump)
        and
        abs(float(max_value) - float(previous_max)) <= float(max_jump)
    )


def _ocr_text_quality(text):
    """
    Small quality score used only to break ties between multiple
    plausible OCR candidates.

    A result containing the decimal point is preferred because the
    camera displays one decimal place.
    """
    text = str(text or "")

    score = 0.0

    if "." in text:
        score += 2.0

    digit_count = sum(
        character.isdigit()
        for character in text
    )

    score += min(digit_count, 4) * 0.25

    return score


def _collect_ocr_candidates(
    frame,
    roi,
    decimal_places,
    primary_method,
    primary_psm,
    padding,
):
    """
    Run the configured OCR method first, then alternative methods.

    Duplicate method/PSM combinations are skipped.

    Returns:
        list of candidate dictionaries
    """
    configurations = [
        (primary_method, int(primary_psm)),
        *OCR_RETRY_CONFIGS,
    ]

    seen = set()
    candidates = []

    for method, psm in configurations:
        key = (method, int(psm))

        if key in seen:
            continue

        seen.add(key)

        value, text, prepared = ocr_temperature(
            frame,
            roi,
            decimal_places=decimal_places,
            method=method,
            psm=psm,
            padding=padding,
        )

        candidates.append({
            "value": value,
            "text": text,
            "method": method,
            "psm": int(psm),
            "prepared": prepared,
            "is_primary": (
                method == primary_method
                and int(psm) == int(primary_psm)
            ),
        })

    return candidates


def _choose_temporally_plausible_pair(
    min_candidates,
    max_candidates,
    previous_min,
    previous_max,
    min_allowed,
    max_allowed,
    min_scale_span,
    max_jump,
):
    """
    Choose the best MIN/MAX pair among all OCR candidates.

    Preference order:
      1) Pair must be physically valid.
      2) Pair must be close to the previous trusted scale.
      3) Pair closest to the previous trusted values wins.
      4) Decimal-looking OCR text is used only as a tie-breaker.
    """
    valid_pairs = []

    for min_candidate in min_candidates:
        min_value = min_candidate["value"]

        if not np.isfinite(min_value):
            continue

        for max_candidate in max_candidates:
            max_value = max_candidate["value"]

            if not _basic_scale_pair_valid(
                min_value=min_value,
                max_value=max_value,
                min_allowed=min_allowed,
                max_allowed=max_allowed,
                min_scale_span=min_scale_span,
            ):
                continue

            if not _temporal_scale_pair_valid(
                min_value=min_value,
                max_value=max_value,
                previous_min=previous_min,
                previous_max=previous_max,
                max_jump=max_jump,
            ):
                continue

            # Primary goal is temporal closeness.
            if previous_min is None or previous_max is None:
                distance = 0.0
            else:
                distance = (
                    abs(float(min_value) - float(previous_min))
                    +
                    abs(float(max_value) - float(previous_max))
                )

            quality = (
                _ocr_text_quality(min_candidate["text"])
                +
                _ocr_text_quality(max_candidate["text"])
            )

            # Smaller tuple is better.
            #
            # - distance dominates
            # - negative quality means higher quality wins ties
            # - primary OCR receives a tiny preference if equally good
            primary_penalty = (
                0
                if (
                    min_candidate["is_primary"]
                    and max_candidate["is_primary"]
                )
                else 1
            )

            score = (
                distance,
                -quality,
                primary_penalty,
            )

            valid_pairs.append(
                (
                    score,
                    min_candidate,
                    max_candidate,
                )
            )

    if not valid_pairs:
        return None

    valid_pairs.sort(
        key=lambda item: item[0]
    )

    _, best_min, best_max = valid_pairs[0]

    return best_min, best_max


def scan_dynamic_scale(
    video_path,
    scale_config,
    output_dir,
    start_frame=0,
    end_frame=None,
    ocr_every=1,
    min_allowed=-100.0,
    max_allowed=1000.0,
    min_scale_span=5.0,
    max_jump=2.5,
    debug_invalid_limit=30,
):
    """
    FIRST PASS through the video.

    Robust dynamic-scale strategy
    -----------------------------
    1. Start from the correctly configured reference-frame scale.
    2. OCR approximately a few times per second.
    3. Accept a reading only if it is physically valid AND close to
       the last trusted MIN/MAX values.
    4. If the configured OCR method gives an implausible result,
       retry THE SAME FRAME with several alternative OCR pipelines.
    5. If every OCR attempt is still unreliable, HOLD the previous
       trusted scale instead of allowing a fake spike such as 467 C.
    6. Frames between OCR sampling points also use the latest trusted
       scale.

    This means a single OCR failure cannot ruin ROI averages/maxima.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if end_frame is None:
        end_frame = total_frames - 1

    start_frame = max(
        0,
        int(start_frame),
    )

    end_frame = min(
        int(end_frame),
        total_frames - 1,
    )

    if end_frame < start_frame:
        cap.release()
        raise ValueError(
            "end_frame must be >= start_frame"
        )

    frame_count = (
        end_frame - start_frame + 1
    )

    # --------------------------------------------------------
    # Arrays saved for audit/debugging.
    # --------------------------------------------------------

    primary_min = np.full(
        frame_count,
        np.nan,
        dtype=np.float64,
    )

    primary_max = np.full(
        frame_count,
        np.nan,
        dtype=np.float64,
    )

    accepted_sample_min = np.full(
        frame_count,
        np.nan,
        dtype=np.float64,
    )

    accepted_sample_max = np.full(
        frame_count,
        np.nan,
        dtype=np.float64,
    )

    # Clean values exist for EVERY video frame.
    clean_min = np.full(
        frame_count,
        np.nan,
        dtype=np.float64,
    )

    clean_max = np.full(
        frame_count,
        np.nan,
        dtype=np.float64,
    )

    primary_min_text = [""] * frame_count
    primary_max_text = [""] * frame_count

    accepted_min_text = [""] * frame_count
    accepted_max_text = [""] * frame_count

    accepted_min_method = [""] * frame_count
    accepted_max_method = [""] * frame_count

    status = ["UNSAMPLED"] * frame_count

    retry_attempted = np.zeros(
        frame_count,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # OCR configuration.
    # --------------------------------------------------------

    min_roi = (
        scale_config["scale"]["min_text_roi"]
    )

    max_roi = (
        scale_config["scale"]["max_text_roi"]
    )

    decimal_places = int(
        scale_config["scale"].get(
            "display_decimal_places",
            1,
        )
    )

    min_ocr_method = (
        scale_config["scale"].get(
            "min_ocr_method",
            "green_difference",
        )
    )

    max_ocr_method = (
        scale_config["scale"].get(
            "max_ocr_method",
            "green_difference",
        )
    )

    min_ocr_psm = int(
        scale_config["scale"].get(
            "min_ocr_psm",
            7,
        )
    )

    max_ocr_psm = int(
        scale_config["scale"].get(
            "max_ocr_psm",
            7,
        )
    )

    ocr_padding = int(
        scale_config["scale"].get(
            "ocr_padding_pixels",
            6,
        )
    )

    # --------------------------------------------------------
    # Trusted starting scale from configuration.
    #
    # configure_thermal_scale_v2.py already validated this exact
    # reference frame using multiple OCR methods, so it is a much
    # stronger starting point than an arbitrary first video OCR.
    # --------------------------------------------------------

    reference_min = (
        scale_config["scale"].get(
            "reference_ocr_min_temp"
        )
    )

    reference_max = (
        scale_config["scale"].get(
            "reference_ocr_max_temp"
        )
    )

    if (
        reference_min is not None
        and reference_max is not None
        and _basic_scale_pair_valid(
            min_value=float(reference_min),
            max_value=float(reference_max),
            min_allowed=min_allowed,
            max_allowed=max_allowed,
            min_scale_span=min_scale_span,
        )
    ):
        last_good_min = float(
            reference_min
        )

        last_good_max = float(
            reference_max
        )

        print(
            "\nTrusted scale seed from configuration:"
            f"\n  MIN = {last_good_min:.1f} C"
            f"\n  MAX = {last_good_max:.1f} C"
        )

    else:
        last_good_min = None
        last_good_max = None

        print(
            "\nWarning: scale_config.json does not contain "
            "a usable trusted reference MIN/MAX."
            "\nThe first fully valid OCR pair will initialize "
            "the temporal filter."
        )

    print(
        "\nConfigured primary OCR:"
        f"\n  MIN = {min_ocr_method}/PSM{min_ocr_psm}"
        f"\n  MAX = {max_ocr_method}/PSM{max_ocr_psm}"
        f"\nMaximum accepted change per OCR sample: "
        f"{float(max_jump):.2f} C"
    )

    # --------------------------------------------------------
    # Debug folder for frames where primary OCR was rejected.
    # --------------------------------------------------------

    invalid_debug_dir = (
        output_dir
        / "ocr_debug_rejected"
    )

    invalid_debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    invalid_saved = 0

    # --------------------------------------------------------
    # Counters shown after PASS 1.
    # --------------------------------------------------------

    counters = {
        "sampled": 0,
        "accepted_primary": 0,
        "accepted_retry": 0,
        "held_previous": 0,
        "initialized_without_reference": 0,
    }

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    print(
        "\nPASS 1/2: Reading + validating dynamic scale labels..."
    )

    print(
        f"Frames: {start_frame} to {end_frame}"
    )

    print(
        f"OCR every {ocr_every} frame(s)"
    )

    for local_idx in range(frame_count):
        frame_number = (
            start_frame + local_idx
        )

        ok, frame = cap.read()

        if not ok:
            print(
                f"\nWarning: could not read frame "
                f"{frame_number}"
            )
            break

        should_ocr = (
            local_idx
            % max(1, int(ocr_every))
            == 0
            or frame_number == end_frame
        )

        if should_ocr:
            counters["sampled"] += 1

            # ------------------------------------------------
            # PRIMARY OCR
            # ------------------------------------------------

            min_value, min_text, min_prepared = (
                ocr_temperature(
                    frame,
                    min_roi,
                    decimal_places=decimal_places,
                    method=min_ocr_method,
                    psm=min_ocr_psm,
                    padding=ocr_padding,
                )
            )

            max_value, max_text, max_prepared = (
                ocr_temperature(
                    frame,
                    max_roi,
                    decimal_places=decimal_places,
                    method=max_ocr_method,
                    psm=max_ocr_psm,
                    padding=ocr_padding,
                )
            )

            primary_min[
                local_idx
            ] = min_value

            primary_max[
                local_idx
            ] = max_value

            primary_min_text[
                local_idx
            ] = min_text

            primary_max_text[
                local_idx
            ] = max_text

            primary_basic_valid = (
                _basic_scale_pair_valid(
                    min_value=min_value,
                    max_value=max_value,
                    min_allowed=min_allowed,
                    max_allowed=max_allowed,
                    min_scale_span=min_scale_span,
                )
            )

            primary_temporal_valid = (
                primary_basic_valid
                and
                _temporal_scale_pair_valid(
                    min_value=min_value,
                    max_value=max_value,
                    previous_min=last_good_min,
                    previous_max=last_good_max,
                    max_jump=max_jump,
                )
            )

            # ------------------------------------------------
            # CASE A: primary OCR is already trustworthy.
            # ------------------------------------------------

            if primary_temporal_valid:
                chosen_min = {
                    "value": float(min_value),
                    "text": min_text,
                    "method": min_ocr_method,
                    "psm": min_ocr_psm,
                    "is_primary": True,
                }

                chosen_max = {
                    "value": float(max_value),
                    "text": max_text,
                    "method": max_ocr_method,
                    "psm": max_ocr_psm,
                    "is_primary": True,
                }

                sample_status = (
                    "ACCEPT_PRIMARY"
                )

                counters[
                    "accepted_primary"
                ] += 1

            # ------------------------------------------------
            # CASE B: primary OCR is bad/suspicious.
            #
            # Retry this SAME frame using multiple alternate OCR
            # preprocessing methods.
            # ------------------------------------------------

            else:
                retry_attempted[
                    local_idx
                ] = 1

                min_candidates = (
                    _collect_ocr_candidates(
                        frame=frame,
                        roi=min_roi,
                        decimal_places=decimal_places,
                        primary_method=min_ocr_method,
                        primary_psm=min_ocr_psm,
                        padding=ocr_padding,
                    )
                )

                max_candidates = (
                    _collect_ocr_candidates(
                        frame=frame,
                        roi=max_roi,
                        decimal_places=decimal_places,
                        primary_method=max_ocr_method,
                        primary_psm=max_ocr_psm,
                        padding=ocr_padding,
                    )
                )

                best_pair = (
                    _choose_temporally_plausible_pair(
                        min_candidates=min_candidates,
                        max_candidates=max_candidates,
                        previous_min=last_good_min,
                        previous_max=last_good_max,
                        min_allowed=min_allowed,
                        max_allowed=max_allowed,
                        min_scale_span=min_scale_span,
                        max_jump=max_jump,
                    )
                )

                if best_pair is not None:
                    chosen_min, chosen_max = (
                        best_pair
                    )

                    sample_status = (
                        "ACCEPT_RETRY"
                    )

                    counters[
                        "accepted_retry"
                    ] += 1

                else:
                    chosen_min = None
                    chosen_max = None

                    # ----------------------------------------
                    # No trustworthy OCR candidate.
                    #
                    # HOLD the last trusted scale rather than
                    # accepting a catastrophic spike.
                    # ----------------------------------------

                    if (
                        last_good_min is not None
                        and last_good_max is not None
                    ):
                        sample_status = (
                            "HOLD_PREVIOUS"
                        )

                        counters[
                            "held_previous"
                        ] += 1

                    else:
                        # We have no history AND no valid OCR.
                        # Refuse to guess.
                        cap.release()

                        raise RuntimeError(
                            "The first sampled frame could not "
                            "produce a trustworthy scale and "
                            "there is no trusted reference value "
                            "in scale_config.json."
                        )

            # ------------------------------------------------
            # Commit accepted sample.
            # ------------------------------------------------

            if chosen_min is not None:
                last_good_min = float(
                    chosen_min["value"]
                )

                last_good_max = float(
                    chosen_max["value"]
                )

                accepted_sample_min[
                    local_idx
                ] = last_good_min

                accepted_sample_max[
                    local_idx
                ] = last_good_max

                accepted_min_text[
                    local_idx
                ] = chosen_min["text"]

                accepted_max_text[
                    local_idx
                ] = chosen_max["text"]

                accepted_min_method[
                    local_idx
                ] = (
                    f"{chosen_min['method']}"
                    f"/PSM{chosen_min['psm']}"
                )

                accepted_max_method[
                    local_idx
                ] = (
                    f"{chosen_max['method']}"
                    f"/PSM{chosen_max['psm']}"
                )

            status[
                local_idx
            ] = sample_status

            # ------------------------------------------------
            # Save rejected-primary diagnostics.
            #
            # Even if retry later succeeded, the primary OCR was
            # suspicious, so keeping a few examples is useful.
            # ------------------------------------------------

            if (
                not primary_temporal_valid
                and invalid_saved
                < debug_invalid_limit
            ):
                prefix = (
                    f"frame_{frame_number:06d}"
                )

                if min_prepared is not None:
                    cv2.imwrite(
                        str(
                            invalid_debug_dir
                            / f"{prefix}_PRIMARY_MIN.png"
                        ),
                        min_prepared,
                    )

                if max_prepared is not None:
                    cv2.imwrite(
                        str(
                            invalid_debug_dir
                            / f"{prefix}_PRIMARY_MAX.png"
                        ),
                        max_prepared,
                    )

                # Also save the original frame text crops.
                min_original = (
                    crop_roi_with_padding(
                        frame,
                        min_roi,
                        padding=ocr_padding,
                    )
                )

                max_original = (
                    crop_roi_with_padding(
                        frame,
                        max_roi,
                        padding=ocr_padding,
                    )
                )

                cv2.imwrite(
                    str(
                        invalid_debug_dir
                        / f"{prefix}_ORIGINAL_MIN.png"
                    ),
                    min_original,
                )

                cv2.imwrite(
                    str(
                        invalid_debug_dir
                        / f"{prefix}_ORIGINAL_MAX.png"
                    ),
                    max_original,
                )

                invalid_saved += 1

        # ----------------------------------------------------
        # Every single frame gets the most recent trusted scale.
        #
        # This is what keeps output duration/FPS independent from
        # the slower OCR sampling rate.
        # ----------------------------------------------------

        if (
            last_good_min is None
            or last_good_max is None
        ):
            cap.release()

            raise RuntimeError(
                "No trustworthy thermal scale is available "
                f"for frame {frame_number}."
            )

        clean_min[
            local_idx
        ] = last_good_min

        clean_max[
            local_idx
        ] = last_good_max

        if (
            local_idx % 100 == 0
            or local_idx == frame_count - 1
        ):
            pct = (
                100.0
                * (local_idx + 1)
                / frame_count
            )

            print(
                f"\rScale OCR: "
                f"{local_idx + 1}/{frame_count} "
                f"({pct:.1f}%)",
                end="",
                flush=True,
            )

    cap.release()
    print()

    # --------------------------------------------------------
    # Final sanity check.
    # --------------------------------------------------------

    if np.any(
        clean_max <= clean_min
    ):
        raise RuntimeError(
            "Some cleaned frames have max_temp <= min_temp. "
            "Check scale OCR results before proceeding."
        )

    # --------------------------------------------------------
    # Save COMPLETE OCR audit trail.
    # --------------------------------------------------------

    scale_csv_path = (
        output_dir
        / "scale_readings.csv"
    )

    with open(
        scale_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "frame",
            "timestamp_seconds",
            "was_ocr_sampled",

            "primary_min_temp",
            "primary_max_temp",
            "primary_min_text",
            "primary_max_text",

            "accepted_sample_min_temp",
            "accepted_sample_max_temp",
            "accepted_min_text",
            "accepted_max_text",
            "accepted_min_method",
            "accepted_max_method",

            "clean_min_temp_used",
            "clean_max_temp_used",

            "status",
            "retry_attempted",
        ])

        for local_idx in range(
            frame_count
        ):
            frame_number = (
                start_frame + local_idx
            )

            writer.writerow([
                frame_number,
                (
                    frame_number / fps
                    if fps > 0
                    else ""
                ),
                int(
                    status[local_idx]
                    != "UNSAMPLED"
                ),

                (
                    ""
                    if not np.isfinite(
                        primary_min[local_idx]
                    )
                    else primary_min[
                        local_idx
                    ]
                ),

                (
                    ""
                    if not np.isfinite(
                        primary_max[local_idx]
                    )
                    else primary_max[
                        local_idx
                    ]
                ),

                primary_min_text[
                    local_idx
                ],

                primary_max_text[
                    local_idx
                ],

                (
                    ""
                    if not np.isfinite(
                        accepted_sample_min[
                            local_idx
                        ]
                    )
                    else accepted_sample_min[
                        local_idx
                    ]
                ),

                (
                    ""
                    if not np.isfinite(
                        accepted_sample_max[
                            local_idx
                        ]
                    )
                    else accepted_sample_max[
                        local_idx
                    ]
                ),

                accepted_min_text[
                    local_idx
                ],

                accepted_max_text[
                    local_idx
                ],

                accepted_min_method[
                    local_idx
                ],

                accepted_max_method[
                    local_idx
                ],

                clean_min[
                    local_idx
                ],

                clean_max[
                    local_idx
                ],

                status[
                    local_idx
                ],

                int(
                    retry_attempted[
                        local_idx
                    ]
                ),
            ])

    # --------------------------------------------------------
    # OCR reliability summary.
    # --------------------------------------------------------

    sampled = counters["sampled"]

    accepted_primary = (
        counters["accepted_primary"]
    )

    accepted_retry = (
        counters["accepted_retry"]
    )

    held_previous = (
        counters["held_previous"]
    )

    accepted_total = (
        accepted_primary
        + accepted_retry
    )

    print("\nOCR VALIDATION SUMMARY")
    print("-" * 52)
    print(
        f"OCR samples attempted : {sampled}"
    )
    print(
        f"Accepted directly     : "
        f"{accepted_primary}"
    )
    print(
        f"Accepted after retry  : "
        f"{accepted_retry}"
    )
    print(
        f"Held previous scale   : "
        f"{held_previous}"
    )

    if sampled > 0:
        print(
            f"Accepted OCR rate     : "
            f"{100.0 * accepted_total / sampled:.1f}%"
        )

    print(
        f"Scale audit CSV       : "
        f"{scale_csv_path}"
    )

    if invalid_saved > 0:
        print(
            f"Rejected OCR examples : "
            f"{invalid_debug_dir}"
        )

    return {
        "fps": fps,
        "total_frames": total_frames,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "clean_min": clean_min,
        "clean_max": clean_max,

        # Primary raw OCR arrays are returned for debugging only.
        "raw_min": primary_min,
        "raw_max": primary_max,

        "status": status,
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
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Stage-3 output ROOT folder. A per-video folder is "
            "created automatically inside it."
        ),
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
        default=None,
        help=(
            "Advanced override: OCR every N frames. "
            "If omitted, --ocr-hz is used instead."
        ),
    )

    parser.add_argument(
        "--ocr-hz",
        type=float,
        default=3.0,
        help=(
            "How many times per second to read the changing scale "
            "with OCR. Default: 3.0. The output video is STILL "
            "written at the original FPS; this controls only OCR."
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
        "--min-scale-span",
        type=float,
        default=5.0,
        help=(
            "Reject OCR MIN/MAX pairs whose difference is smaller "
            "than this many degrees C. Default 5.0. This prevents "
            "bad pairs such as 0.1 C to 0.8 C from being accepted."
        ),
    )

    parser.add_argument(
        "--max-scale-jump",
        type=float,
        default=2.5,
        help=(
            "Maximum allowed change in either scale MIN or MAX "
            "between consecutive OCR samples. Default: 2.5 C. "
            "If primary OCR exceeds this, the same frame is retried "
            "with alternate OCR methods. If all retries still fail, "
            "the previous trusted scale is held."
        ),
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

    # Keep the output hierarchy aligned with raw-videos.
    #
    # Example:
    #   raw-videos/Furnace/video.mp4
    #       ->
    #   step03-roi-videos/Furnace/video/
    output_dir = build_video_output_dir(
        video_path=video_path,
        output_root=args.output,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nStage-3 output folder:\n{output_dir}"
    )

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
    fps = float(cap.get(cv2.CAP_PROP_FPS))
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
    # Decide OCR cadence.
    #
    # IMPORTANT:
    # OCR frequency and output-video FPS are separate things.
    #
    # Example at 25 FPS:
    #   --ocr-hz 3
    #       -> OCR approximately every 8 frames
    #       -> temperature/polygon calculations still happen
    #          on ALL 25 frames each second
    #       -> output video remains 25 FPS and full duration
    # --------------------------------------------------------

    if args.ocr_every is not None:
        ocr_every = max(
            1,
            int(args.ocr_every),
        )
    else:
        if fps <= 0:
            raise RuntimeError(
                "Video reported invalid FPS, so --ocr-hz "
                "cannot be converted to a frame interval."
            )

        if float(args.ocr_hz) <= 0:
            raise ValueError(
                "--ocr-hz must be greater than 0."
            )

        ocr_every = max(
            1,
            int(round(fps / float(args.ocr_hz))),
        )

    effective_ocr_hz = (
        fps / ocr_every
        if fps > 0
        else 0.0
    )

    print(
        f"\nVideo FPS: {fps:.3f}"
        f"\nOCR interval: every {ocr_every} frame(s)"
        f"\nEffective OCR rate: {effective_ocr_hz:.3f} reads/second"
    )

    if args.max_frames is not None:
        test_frames = end_frame - start_frame + 1

        test_duration = (
            test_frames / fps
            if fps > 0
            else 0.0
        )

        print(
            "\nTEST MODE IS ACTIVE because --max-frames was supplied."
            f"\nOnly {test_frames} frames will be written."
            f"\nExpected output duration: approximately "
            f"{test_duration:.2f} seconds."
            "\nRemove --max-frames for a full-length output video."
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
        ocr_every=ocr_every,
        min_allowed=float(args.min_allowed_temp),
        max_allowed=float(args.max_allowed_temp),
        min_scale_span=float(args.min_scale_span),
        max_jump=max(
            0.0,
            float(args.max_scale_jump),
        ),
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
