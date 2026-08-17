import argparse
import cv2
import json
import pickle
from pathlib import Path


# ============================================================
# HELPERS
# ============================================================

def select_roi(window_title, frame):
    """
    Let the user drag a rectangle on the ORIGINAL frame.

    cv2.selectROI returns:
        (x, y, width, height)

    Press ENTER or SPACE to confirm a selection.
    Press C while selecting to cancel/reselect.
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
        raise RuntimeError(f"No valid ROI selected for: {window_title}")

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


# ============================================================
# MAIN CONFIGURATION WORKFLOW
# ============================================================

def configure(video_path, regions_json_path, output_dir, frame_override=None):
    video_path = Path(video_path)
    regions_json_path = Path(regions_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Read the annotation JSON so we can reuse the same
    # reference frame that was used for equipment annotation.
    # --------------------------------------------------------
    with open(regions_json_path, "r", encoding="utf-8") as f:
        regions_data = json.load(f)

    reference_frame_number = int(
        regions_data.get("video", {}).get("reference_frame_number", 0)
    )

    if frame_override is not None:
        reference_frame_number = int(frame_override)

    # --------------------------------------------------------
    # Open the source video and grab the reference frame.
    # --------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if total_frames <= 0:
        raise RuntimeError("Video reports zero frames.")

    reference_frame_number = max(
        0,
        min(reference_frame_number, total_frames - 1),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_number)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(
            f"Could not read reference frame {reference_frame_number}."
        )

    height, width = frame.shape[:2]

    print("\n============================================")
    print("THERMAL SCALE CONFIGURATION")
    print("============================================")
    print(f"Video: {video_path.name}")
    print(f"Resolution: {width} x {height}")
    print(f"Reference frame: {reference_frame_number}")
    if fps > 0:
        print(f"Reference time: {reference_frame_number / fps:.2f} s")

    print("\nYou will select THREE rectangles.")
    print("1) The vertical grayscale/color bar ONLY")
    print("2) The numeric MAX temperature text")
    print("3) The numeric MIN temperature text")
    print("\nKeep the text crops tight around the digits if possible.")
    print("For example, include '46.7 C' but avoid unrelated UI text.\n")

    # --------------------------------------------------------
    # Select the actual vertical palette/color bar.
    # Do NOT include max/min text inside the bar rectangle.
    # --------------------------------------------------------
    print("SELECT 1/3: COLOR BAR")
    bar_roi = select_roi(
        "1 - Select THERMAL COLOR BAR, then press ENTER",
        frame,
    )

    # --------------------------------------------------------
    # Select the text showing the maximum scale temperature.
    # --------------------------------------------------------
    print("SELECT 2/3: MAX TEMPERATURE TEXT")
    max_text_roi = select_roi(
        "2 - Select MAX temperature text, then press ENTER",
        frame,
    )

    # --------------------------------------------------------
    # Select the text showing the minimum scale temperature.
    # --------------------------------------------------------
    print("SELECT 3/3: MIN TEMPERATURE TEXT")
    min_text_roi = select_roi(
        "3 - Select MIN temperature text, then press ENTER",
        frame,
    )

    # --------------------------------------------------------
    # Save both JSON and PKL, matching the annotation workflow.
    # --------------------------------------------------------
    config = {
        "video": {
            "filename": video_path.name,
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "reference_frame_number": reference_frame_number,
        },
        "scale": {
            "bar_roi": bar_roi,
            "max_text_roi": max_text_roi,
            "min_text_roi": min_text_roi,

            # When sampling the color bar later, the processor
            # ignores the left/right 20% of the selected bar.
            # This helps avoid borders, ticks, or compression noise.
            "bar_horizontal_inner_fraction": 0.20,

            # The thermal camera in the provided example displays
            # one decimal place, e.g. 46.7 C and 15.4 C.
            "display_decimal_places": 1,
        },
    }

    json_path = output_dir / "scale_config.json"
    pkl_path = output_dir / "scale_config.pkl"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    with open(pkl_path, "wb") as f:
        pickle.dump(config, f)

    # --------------------------------------------------------
    # Save a visual preview so the configuration can be checked.
    # --------------------------------------------------------
    preview = frame.copy()

    draw_box(preview, bar_roi, "COLOR BAR", (255, 0, 255))
    draw_box(preview, max_text_roi, "MAX TEXT", (0, 255, 0))
    draw_box(preview, min_text_roi, "MIN TEXT", (0, 255, 255))

    preview_path = output_dir / "scale_config_preview.png"
    cv2.imwrite(str(preview_path), preview)

    print("\nSaved thermal scale configuration:")
    print(f"JSON    : {json_path}")
    print(f"PKL     : {pkl_path}")
    print(f"Preview : {preview_path}")


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Select the thermal color bar and dynamic min/max "
            "temperature label regions for a fixed thermal video."
        )
    )

    parser.add_argument(
        "video",
        help="Path to the ORIGINAL thermal video",
    )

    parser.add_argument(
        "regions_json",
        help="Path to regions.json created by the ROI annotator",
    )

    parser.add_argument(
        "--output",
        default="thermal_scale_config",
        help="Folder for scale_config.json/.pkl and preview",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help=(
            "Optional frame number override. By default this uses "
            "the reference frame stored inside regions.json."
        ),
    )

    args = parser.parse_args()

    configure(
        video_path=args.video,
        regions_json_path=args.regions_json,
        output_dir=args.output,
        frame_override=args.frame,
    )
