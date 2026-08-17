import argparse
import json
import pickle
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract

# Default Stage-2 output root.
DEFAULT_OUTPUT_ROOT = "step02-ocr-scale-config"


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
# OCR METHODS TESTED DURING CONFIGURATION
# ============================================================

# The configurator tries several combinations ONCE on the reference
# frame. The winning preprocessing method + Tesseract PSM are then
# saved into scale_config.json.
#
# During the full video, the processor uses only the selected method,
# so we get robustness without running 5-10 OCR attempts per frame.
OCR_METHODS_TO_TEST = [
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

def select_roi(window_title, frame):
    """
    Drag a rectangle on the ORIGINAL frame.

    Press ENTER or SPACE to confirm.
    """
    roi = cv2.selectROI(
        window_title,
        frame,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(window_title)

    x, y, w, h = [int(v) for v in roi]

    if w <= 0 or h <= 0:
        raise RuntimeError(
            f"No valid ROI selected for: {window_title}"
        )

    return [x, y, w, h]


def draw_box(frame, roi, label, color):
    """Draw one labeled rectangle on a preview image."""
    x, y, w, h = roi

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color,
        2,
    )

    cv2.putText(
        frame,
        label,
        (x, max(25, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def crop_roi_with_padding(frame, roi, padding=6):
    """
    Expand the selected OCR region slightly so a tight selection
    does not clip the first/last digit.
    """
    x, y, w, h = [int(v) for v in roi]
    padding = max(0, int(padding))

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame.shape[1], x + w + padding)
    y2 = min(frame.shape[0], y + h + padding)

    return frame[y1:y2, x1:x2]


def parse_temperature(text, decimal_places=1):
    """
    Safely parse a thermal scale value.

    A one-digit partial OCR result is rejected rather than turned
    into a fake 0.1/0.8 reading.
    """
    if text is None:
        return np.nan

    cleaned = re.sub(
        r"[^0-9.\-]",
        "",
        text.strip(),
    )

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        cleaned,
    )

    if not match:
        return np.nan

    token = match.group(0)

    try:
        if "." in token:
            return float(token)

        if decimal_places > 0:
            sign = (
                -1.0
                if token.startswith("-")
                else 1.0
            )

            digits = token.lstrip("-")

            minimum_digits = (
                decimal_places + 2
            )

            if len(digits) < minimum_digits:
                return np.nan

            return sign * (
                int(digits)
                / (10 ** decimal_places)
            )

        return float(token)

    except ValueError:
        return np.nan


# ============================================================
# OCR PREPROCESSING
# ============================================================

def finish_binary(binary):
    """
    Convert a binary image to black text on white background,
    enlarge it, and add padding.
    """
    if binary is None or binary.size == 0:
        return None

    if float(np.mean(binary)) < 127.0:
        binary = cv2.bitwise_not(
            binary
        )

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


def prepare_crop(crop, method):
    """
    Make one OCR-ready version of the scale-label crop.
    """
    if crop is None or crop.size == 0:
        return None

    b, g, r = cv2.split(crop)

    if method == "green_difference":
        b16 = b.astype(np.int16)
        g16 = g.astype(np.int16)
        r16 = r.astype(np.int16)

        advantage = (
            g16 - np.maximum(r16, b16)
        )

        mask = (
            (advantage >= 4)
            & (g >= 70)
        ).astype(np.uint8) * 255

        kernel = np.ones(
            (2, 2),
            dtype=np.uint8,
        )

        # Text is WHITE foreground here. This is intentional.
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

        return finish_binary(mask)

    if method == "hsv_green":
        hsv = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2HSV,
        )

        mask = cv2.inRange(
            hsv,
            np.array([25, 12, 65], dtype=np.uint8),
            np.array([100, 255, 255], dtype=np.uint8),
        )

        kernel = np.ones(
            (2, 2),
            dtype=np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        return finish_binary(mask)

    if method == "green_channel_otsu":
        channel = cv2.GaussianBlur(
            g,
            (3, 3),
            0,
        )

        _, binary = cv2.threshold(
            channel,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU,
        )

        return finish_binary(binary)

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
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU,
        )

        return finish_binary(binary)

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

        return finish_binary(binary)

    raise ValueError(
        f"Unknown OCR method: {method}"
    )


def run_one_ocr(
    frame,
    roi,
    method,
    psm,
    decimal_places,
    padding,
):
    """Run one preprocessing + Tesseract combination."""
    crop = crop_roi_with_padding(
        frame,
        roi,
        padding=padding,
    )

    prepared = prepare_crop(
        crop,
        method=method,
    )

    config = (
        f"--psm {int(psm)} "
        "-c tessedit_char_whitelist=0123456789.-"
    )

    raw_text = pytesseract.image_to_string(
        prepared,
        config=config,
    ).strip()

    value = parse_temperature(
        raw_text,
        decimal_places=decimal_places,
    )

    return {
        "method": method,
        "psm": int(psm),
        "raw_text": raw_text,
        "value": value,
        "prepared": prepared,
        "original_crop": crop,
    }


def candidate_quality(candidate):
    """
    Give preference to OCR outputs that visibly contain a decimal
    point and contain enough digits to look complete.
    """
    text = candidate["raw_text"]

    digits = sum(
        char.isdigit()
        for char in text
    )

    score = 0.0

    if "." in text:
        score += 4.0

    score += min(
        digits,
        4,
    )

    # Earlier methods in OCR_METHODS_TO_TEST are lightly preferred
    # when quality otherwise ties.
    return score


def choose_best_pair(
    min_candidates,
    max_candidates,
    min_allowed,
    max_allowed,
    min_scale_span,
):
    """
    Select the strongest plausible MIN/MAX pair.

    MIN and MAX are allowed to use different preprocessing methods
    if that is what reads the frame best.
    """
    valid_pairs = []

    for min_candidate in min_candidates:
        min_value = min_candidate["value"]

        if not np.isfinite(min_value):
            continue

        if not (
            min_allowed
            <= min_value
            <= max_allowed
        ):
            continue

        for max_candidate in max_candidates:
            max_value = max_candidate["value"]

            if not np.isfinite(max_value):
                continue

            if not (
                min_allowed
                <= max_value
                <= max_allowed
            ):
                continue

            span = (
                max_value - min_value
            )

            if span < min_scale_span:
                continue

            score = (
                candidate_quality(
                    min_candidate
                )
                + candidate_quality(
                    max_candidate
                )
            )

            # Small bonus if both labels can use the same OCR setup.
            if (
                min_candidate["method"]
                == max_candidate["method"]
                and min_candidate["psm"]
                == max_candidate["psm"]
            ):
                score += 1.0

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
        key=lambda item: item[0],
        reverse=True,
    )

    _, best_min, best_max = (
        valid_pairs[0]
    )

    return best_min, best_max


def save_ocr_diagnostics(
    output_dir,
    min_candidates,
    max_candidates,
):
    """
    Save every prepared reference-frame crop so OCR problems can
    be inspected visually instead of guessed.
    """
    debug_dir = (
        Path(output_dir)
        / "ocr_config_debug"
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for label, candidates in (
        ("MIN", min_candidates),
        ("MAX", max_candidates),
    ):
        for index, candidate in enumerate(
            candidates,
            start=1,
        ):
            method = candidate["method"]
            psm = candidate["psm"]

            cv2.imwrite(
                str(
                    debug_dir
                    / (
                        f"{label}_{index:02d}_"
                        f"{method}_psm{psm}.png"
                    )
                ),
                candidate["prepared"],
            )

        # Original crop once for comparison.
        if candidates:
            cv2.imwrite(
                str(
                    debug_dir
                    / f"{label}_original_crop.png"
                ),
                candidates[0][
                    "original_crop"
                ],
            )

    return debug_dir


# ============================================================
# MAIN CONFIGURATION WORKFLOW
# ============================================================

def configure(
    video_path,
    regions_json_path,
    output_dir,
    frame_override=None,
    tesseract_path=None,
    min_allowed=-100.0,
    max_allowed=1000.0,
    min_scale_span=5.0,
    ocr_padding=6,
):
    video_path = Path(video_path)
    regions_json_path = Path(
        regions_json_path
    )
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = (
            tesseract_path
        )

    # Confirm Tesseract BEFORE the user spends time selecting ROIs.
    try:
        version = (
            pytesseract.get_tesseract_version()
        )

        print(
            f"Tesseract detected: {version}"
        )

    except Exception as exc:
        raise RuntimeError(
            "Tesseract could not be started. "
            "Pass the executable using --tesseract."
        ) from exc

    with open(
        regions_json_path,
        "r",
        encoding="utf-8",
    ) as file:
        regions_data = json.load(file)

    reference_frame_number = int(
        regions_data
        .get("video", {})
        .get("reference_frame_number", 0)
    )

    if frame_override is not None:
        reference_frame_number = int(
            frame_override
        )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(
            "Video reports zero frames."
        )

    reference_frame_number = max(
        0,
        min(
            reference_frame_number,
            total_frames - 1,
        ),
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        reference_frame_number,
    )

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(
            f"Could not read reference frame "
            f"{reference_frame_number}."
        )

    height, width = frame.shape[:2]

    print("\n============================================")
    print("THERMAL SCALE + OCR CONFIGURATION")
    print("============================================")
    print(f"Video: {video_path.name}")
    print(f"Resolution: {width} x {height}")
    print(
        f"Reference frame: "
        f"{reference_frame_number}"
    )

    if fps > 0:
        print(
            f"Reference time: "
            f"{reference_frame_number / fps:.2f} s"
        )

    print("\nYou will select THREE rectangles:")
    print("1) Vertical grayscale/color bar ONLY")
    print("2) MAX temperature text")
    print("3) MIN temperature text")
    print(
        "\nFor MAX/MIN, include the complete number "
        "with a few pixels around it."
    )
    print(
        "Do not crop so tightly that the first digit "
        "touches the selection edge.\n"
    )

    print("SELECT 1/3: COLOR BAR")

    bar_roi = select_roi(
        "1 - Select THERMAL COLOR BAR, then press ENTER",
        frame,
    )

    print("SELECT 2/3: MAX TEMPERATURE TEXT")

    max_text_roi = select_roi(
        "2 - Select MAX temperature text, then press ENTER",
        frame,
    )

    print("SELECT 3/3: MIN TEMPERATURE TEXT")

    min_text_roi = select_roi(
        "3 - Select MIN temperature text, then press ENTER",
        frame,
    )

    decimal_places = 1

    print("\nTesting OCR methods on the reference frame...")

    min_candidates = []
    max_candidates = []

    for method, psm in OCR_METHODS_TO_TEST:
        min_candidates.append(
            run_one_ocr(
                frame,
                min_text_roi,
                method=method,
                psm=psm,
                decimal_places=decimal_places,
                padding=ocr_padding,
            )
        )

        max_candidates.append(
            run_one_ocr(
                frame,
                max_text_roi,
                method=method,
                psm=psm,
                decimal_places=decimal_places,
                padding=ocr_padding,
            )
        )

    debug_dir = save_ocr_diagnostics(
        output_dir,
        min_candidates,
        max_candidates,
    )

    print("\nOCR TEST RESULTS")
    print("-" * 78)
    print(
        f"{'Method':24} {'PSM':>4} "
        f"{'MIN OCR':>14} {'MIN':>8} "
        f"{'MAX OCR':>14} {'MAX':>8}"
    )
    print("-" * 78)

    for min_c, max_c in zip(
        min_candidates,
        max_candidates,
    ):
        min_value = (
            "FAIL"
            if not np.isfinite(
                min_c["value"]
            )
            else f'{min_c["value"]:.1f}'
        )

        max_value = (
            "FAIL"
            if not np.isfinite(
                max_c["value"]
            )
            else f'{max_c["value"]:.1f}'
        )

        print(
            f'{min_c["method"]:24} '
            f'{min_c["psm"]:>4} '
            f'{repr(min_c["raw_text"]):>14} '
            f'{min_value:>8} '
            f'{repr(max_c["raw_text"]):>14} '
            f'{max_value:>8}'
        )

    best_pair = choose_best_pair(
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        min_allowed=float(
            min_allowed
        ),
        max_allowed=float(
            max_allowed
        ),
        min_scale_span=float(
            min_scale_span
        ),
    )

    if best_pair is None:
        print(
            "\nOCR CONFIGURATION FAILED."
        )

        print(
            "No plausible MIN/MAX pair was found."
        )

        print(
            f"Diagnostic crops were saved to:\n"
            f"{debug_dir}"
        )

        print(
            "\nThe script has NOT saved a usable "
            "scale_config.json because guessing would "
            "produce incorrect temperatures."
        )

        print(
            "\nRe-run the configurator and make the "
            "MAX/MIN selections a little wider around "
            "the full numbers."
        )

        raise RuntimeError(
            "Could not configure reliable OCR."
        )

    best_min, best_max = best_pair

    print("\nSELECTED OCR CONFIGURATION")
    print(
        f"MIN: {best_min['value']:.1f} C "
        f"from {repr(best_min['raw_text'])} "
        f"using {best_min['method']} "
        f"PSM {best_min['psm']}"
    )

    print(
        f"MAX: {best_max['value']:.1f} C "
        f"from {repr(best_max['raw_text'])} "
        f"using {best_max['method']} "
        f"PSM {best_max['psm']}"
    )

    print(
        f"Scale span on reference frame: "
        f"{best_max['value'] - best_min['value']:.1f} C"
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {
        "video": {
            "filename": video_path.name,
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "reference_frame_number": (
                reference_frame_number
            ),
        },
        "scale": {
            "bar_roi": bar_roi,
            "max_text_roi": (
                max_text_roi
            ),
            "min_text_roi": (
                min_text_roi
            ),
            "bar_horizontal_inner_fraction": 0.20,
            "display_decimal_places": decimal_places,

            # New OCR configuration.
            "ocr_padding_pixels": int(
                ocr_padding
            ),
            "min_ocr_method": (
                best_min["method"]
            ),
            "min_ocr_psm": int(
                best_min["psm"]
            ),
            "max_ocr_method": (
                best_max["method"]
            ),
            "max_ocr_psm": int(
                best_max["psm"]
            ),

            # Save the reference reading for auditing only.
            "reference_ocr_min_temp": float(
                best_min["value"]
            ),
            "reference_ocr_max_temp": float(
                best_max["value"]
            ),
            "reference_ocr_min_text": (
                best_min["raw_text"]
            ),
            "reference_ocr_max_text": (
                best_max["raw_text"]
            ),

            # Plausibility guard.
            "min_scale_span_c": float(
                min_scale_span
            ),
        },
    }

    json_path = (
        output_dir
        / "scale_config.json"
    )

    pkl_path = (
        output_dir
        / "scale_config.pkl"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=4,
        )

    with open(
        pkl_path,
        "wb",
    ) as file:
        pickle.dump(
            config,
            file,
        )

    preview = frame.copy()

    draw_box(
        preview,
        bar_roi,
        "COLOR BAR",
        (255, 0, 255),
    )

    draw_box(
        preview,
        max_text_roi,
        (
            f"MAX TEXT -> "
            f"{best_max['value']:.1f} C"
        ),
        (0, 255, 0),
    )

    draw_box(
        preview,
        min_text_roi,
        (
            f"MIN TEXT -> "
            f"{best_min['value']:.1f} C"
        ),
        (0, 255, 255),
    )

    preview_path = (
        output_dir
        / "scale_config_preview.png"
    )

    cv2.imwrite(
        str(preview_path),
        preview,
    )

    print("\nConfiguration saved successfully:")
    print(f"JSON       : {json_path}")
    print(f"PKL        : {pkl_path}")
    print(f"Preview    : {preview_path}")
    print(f"OCR debug  : {debug_dir}")


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Select thermal scale regions and automatically "
            "configure Tesseract OCR for the camera labels."
        )
    )

    parser.add_argument(
        "video",
        help=(
            "Path to the ORIGINAL thermal video"
        ),
    )

    parser.add_argument(
        "regions_json",
        help=(
            "Path to regions.json created by "
            "the ROI annotator"
        ),
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Stage-2 output ROOT folder. A per-video folder is "
            "created automatically inside it."
        ),
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help=(
            "Optional reference-frame override"
        ),
    )

    parser.add_argument(
        "--tesseract",
        required=True,
        help=(
            "Full path to tesseract.exe. "
            "This avoids needing Windows PATH."
        ),
    )

    parser.add_argument(
        "--min-allowed-temp",
        type=float,
        default=-100.0,
        help=(
            "Reject reference OCR temperatures "
            "below this value"
        ),
    )

    parser.add_argument(
        "--max-allowed-temp",
        type=float,
        default=1000.0,
        help=(
            "Reject reference OCR temperatures "
            "above this value"
        ),
    )

    parser.add_argument(
        "--min-scale-span",
        type=float,
        default=5.0,
        help=(
            "Minimum plausible MAX-MIN scale "
            "difference in degrees C. Default 5."
        ),
    )

    parser.add_argument(
        "--ocr-padding",
        type=int,
        default=6,
        help=(
            "Pixels automatically added around "
            "the selected MAX/MIN text boxes. "
            "Default 6."
        ),
    )

    args = parser.parse_args()

    video_path = Path(args.video)

    # Keep the output hierarchy aligned with raw-videos.
    #
    # Example:
    #   raw-videos/Furnace/video.mp4
    #       ->
    #   step02-ocr-scale-config/Furnace/video/
    output_dir = build_video_output_dir(
        video_path=video_path,
        output_root=args.output,
    )

    print(
        f"\nStage-2 output folder:\n{output_dir}"
    )

    configure(
        video_path=video_path,
        regions_json_path=args.regions_json,
        output_dir=output_dir,
        frame_override=args.frame,
        tesseract_path=args.tesseract,
        min_allowed=args.min_allowed_temp,
        max_allowed=args.max_allowed_temp,
        min_scale_span=args.min_scale_span,
        ocr_padding=args.ocr_padding,
    )
