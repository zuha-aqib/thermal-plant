import cv2
import json
import pickle
import argparse
import shutil
from pathlib import Path

import numpy as np

import step_code as ps


# ============================================================
# CONFIGURATION
# ============================================================

# Final annotations are drawn in RED.
# OpenCV uses BGR instead of RGB.
RED = (0, 0, 255)

# Current points being selected are shown in yellow so that
# they are visually different from confirmed annotations.
YELLOW = (0, 255, 255)

# Text color.
WHITE = (255, 255, 255)

# Maximum size of the annotation window.
# The original coordinates are preserved even if the frame
# needs to be resized to fit on the monitor.
MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 850

# Thickness of annotation lines.
LINE_THICKNESS = 2

# Default output root requested for this project.
#
# Example:
#
#   raw-videos/
#       95.fur incin AB Stacks Thermal.mp4
#
# becomes:
#
#   step-01-annotate-videos/
#       95.fur incin AB Stacks Thermal/
#           regions.json
#           regions.pkl
#           reference_frame.png
#           annotated_reference_frame.png
#           annotated_video.mp4
#
DEFAULT_OUTPUT_ROOT = "step-01-annotate-videos"

# Base files prove that the old Step-01 workflow finished.
# The manifest is stored separately so pre-manifest/legacy outputs can
# still be recognized instead of being misclassified as partial.
REQUIRED_BASE_FILES = (
    "regions.json",
    "regions.pkl",
    "annotated_video.mp4",
)

MANIFEST_FILENAME = "stage_01_manifest.json"


# ============================================================
# GLOBAL ANNOTATION STATE
# ============================================================

# These are reset before EVERY video. This is important for
# batch processing, otherwise annotations from one video could
# accidentally carry over to the next video.
current_points = []
regions = []

# Default drawing mode.
drawing_mode = "polygon"

# Used to map resized GUI coordinates back to original pixels.
display_scale = 1.0

# Original frame used for annotation.
original_frame = None


# ============================================================
# BASIC PATH / BATCH HELPERS
# ============================================================

def is_mp4_file(path):
    """Return True for .mp4 files, case-insensitively."""
    return path.is_file() and path.suffix.lower() == ".mp4"


def discover_mp4_videos(input_path):
    """
    Accept either:
      1. a single MP4 file, or
      2. a directory containing MP4 files at any nesting depth.

    Directory searches are recursive, so this works:

        raw-videos/
            plant-a/
                camera-1/
                    video1.mp4
            plant-b/
                video2.mp4
    """

    input_path = Path(input_path)

    if input_path.is_file():
        if not is_mp4_file(input_path):
            raise ValueError(
                f"Input file is not an MP4 video: {input_path}"
            )

        return [input_path]

    if input_path.is_dir():
        videos = [
            path
            for path in input_path.rglob("*")
            if is_mp4_file(path)
        ]

        # Stable alphabetical order makes batch runs predictable.
        videos.sort(key=lambda p: str(p).lower())

        return videos

    raise FileNotFoundError(
        f"Input path does not exist: {input_path}"
    )


def find_raw_videos_ancestor(video_path):
    """
    When a single video is passed directly, try to find a parent
    directory literally named 'raw-videos'.

    This lets us preserve nested source folders even when the
    command points to one individual file.

    Example:

        raw-videos/Area-A/Camera-01/video.mp4

    becomes:

        step-01-annotate-videos/Area-A/Camera-01/video/
    """

    for parent in video_path.parents:
        if parent.name.lower() == "raw-videos":
            return parent

    return None


def build_output_dir(video_path, input_path, output_root):
    """
    Build the final output directory for one video.

    Rules
    -----
    Folder input:
        Preserve every folder underneath the folder that the user passed.

        raw-videos/Area-A/Camera-01/video.mp4
            ->
        step-01-annotate-videos/Area-A/Camera-01/video/

    Single-file input:
        If the video is somewhere under a folder named raw-videos,
        preserve the folders below raw-videos.

        Otherwise:
            step-01-annotate-videos/video/
    """

    video_path = Path(video_path).resolve()
    input_path = Path(input_path).resolve()
    output_root = Path(output_root).resolve()

    if input_path.is_dir():
        relative_video = video_path.relative_to(input_path)
        relative_parent = relative_video.parent

    else:
        raw_root = find_raw_videos_ancestor(video_path)

        if raw_root is not None:
            relative_video = video_path.relative_to(raw_root)
            relative_parent = relative_video.parent
        else:
            relative_parent = Path()

    return output_root / relative_parent / video_path.stem


def file_exists_and_nonempty(path):
    """Delegate the shared non-empty-file check to step_code.py."""
    return ps.file_exists_and_nonempty(path)


def is_annotation_base_complete(output_dir):
    """True for both legacy and tracked completed Step-01 outputs."""
    return ps.verify_required_files(
        output_dir,
        REQUIRED_BASE_FILES,
    )


def is_annotation_complete(output_dir):
    """True only when base outputs AND the provenance manifest exist."""
    output_dir = Path(output_dir)

    return (
        is_annotation_base_complete(output_dir)
        and file_exists_and_nonempty(
            output_dir / MANIFEST_FILENAME
        )
    )


def has_any_output(output_dir):
    """Use the shared output-folder check."""
    return ps.has_any_output(output_dir)


def prompt_yes_no(message, default=True):
    """Use one consistent prompt implementation across all stages."""
    return ps.prompt_yes_no(message, default=default)


def safe_remove_directory(path):
    """Use the shared safe-removal helper."""
    ps.safe_remove(path)


def finalize_completed_output(working_dir, final_dir):
    """Use the shared safe promotion/backup logic."""
    ps.promote_completed_directory(working_dir, final_dir)


def resolve_reference_frame(video_path, requested_frame):
    """Clamp a requested frame exactly the same way annotation does."""
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    if total_frames <= 0:
        raise RuntimeError(f"Video reported zero frames: {video_path}")

    return max(0, min(int(requested_frame), total_frames - 1))


def build_stage_01_manifest(video_path, working_output_dir):
    """Create the provenance receipt for one completed annotation run."""
    working_output_dir = Path(working_output_dir)

    regions_path = working_output_dir / "regions.json"
    regions_data = ps.load_json(regions_path)

    reference_frame = int(
        regions_data.get("video", {}).get("reference_frame_number", 0)
    )

    source_fingerprint = ps.fingerprint_video(video_path)

    return ps.make_manifest(
        "step_01",
        source_video=source_fingerprint,
        inputs={},
        settings={
            "reference_frame_number": reference_frame,
        },
        outputs={
            "regions_json_sha256": ps.hash_json_file(regions_path),
            "regions_pkl_filename": "regions.pkl",
            "annotated_video_filename": "annotated_video.mp4",
        },
    )


def step_01_stale_reasons(
    existing_manifest,
    current_source,
    desired_reference_frame,
    output_dir,
):
    """Explain why an existing tracked Step-01 result is no longer current."""
    reasons = []

    old_source = existing_manifest.get("source_video", {})

    if not ps.same_file_content(old_source, current_source):
        reasons.append("Source/raw video content changed")

    old_reference = existing_manifest.get("settings", {}).get(
        "reference_frame_number"
    )

    if old_reference != int(desired_reference_frame):
        reasons.append(
            f"Requested reference frame changed ({old_reference} -> {desired_reference_frame})"
        )

    # If somebody manually edits regions.json after Step 01 completed, the
    # old annotated_video.mp4 no longer represents those coordinates.
    regions_path = Path(output_dir) / "regions.json"
    stored_regions_hash = existing_manifest.get("outputs", {}).get(
        "regions_json_sha256"
    )

    if ps.file_exists_and_nonempty(regions_path):
        current_regions_hash = ps.hash_json_file(regions_path)

        if stored_regions_hash != current_regions_hash:
            reasons.append(
                "regions.json changed after the tracked Step-01 run "
                "(annotated_video.mp4 is based on older coordinates)"
            )

    return reasons


# ============================================================
# DISPLAY / COORDINATE HELPERS
# ============================================================

def calculate_display_scale(width, height):
    """
    Calculate how much the original frame needs to be scaled
    so that it fits comfortably on the screen.

    This ONLY affects display size. Saved coordinates stay in
    original video pixel coordinates.
    """

    width_scale = MAX_DISPLAY_WIDTH / width
    height_scale = MAX_DISPLAY_HEIGHT / height

    # Never enlarge beyond original resolution.
    return min(width_scale, height_scale, 1.0)


def display_to_original(x, y):
    """
    Convert a mouse click on the resized preview back into the
    original full-resolution video coordinate system.
    """

    original_x = int(round(x / display_scale))
    original_y = int(round(y / display_scale))

    original_x = max(
        0,
        min(original_x, original_frame.shape[1] - 1)
    )

    original_y = max(
        0,
        min(original_y, original_frame.shape[0] - 1)
    )

    return original_x, original_y


def rectangle_points(point1, point2):
    """
    Convert two opposite rectangle corners into four polygon
    points.

    Saving both rectangles and polygons as a list of final
    vertices makes later mask generation simpler.
    """

    x1, y1 = point1
    x2, y2 = point2

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


# ============================================================
# DRAW ANNOTATION PREVIEW
# ============================================================

def draw_annotation_preview():
    """
    Build the current GUI preview.

    Confirmed regions:
        RED

    Current unfinished points:
        YELLOW

    The underlying original_frame is never modified.
    """

    frame = original_frame.copy()

    # --------------------------------------------------------
    # Draw confirmed ROIs
    # --------------------------------------------------------

    for region in regions:
        points = np.array(
            region["pixel_points"],
            dtype=np.int32
        )

        cv2.polylines(
            frame,
            [points],
            isClosed=True,
            color=RED,
            thickness=LINE_THICKNESS
        )

        min_x = int(np.min(points[:, 0]))
        min_y = int(np.min(points[:, 1]))

        cv2.putText(
            frame,
            region["name"],
            (min_x, max(25, min_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            RED,
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Draw current unfinished annotation
    # --------------------------------------------------------

    if current_points:
        for point in current_points:
            cv2.circle(
                frame,
                tuple(point),
                radius=5,
                color=YELLOW,
                thickness=-1
            )

        if drawing_mode == "polygon":
            if len(current_points) >= 2:
                points = np.array(
                    current_points,
                    dtype=np.int32
                )

                cv2.polylines(
                    frame,
                    [points],
                    isClosed=False,
                    color=YELLOW,
                    thickness=LINE_THICKNESS
                )

        elif drawing_mode == "rectangle":
            if len(current_points) == 2:
                cv2.rectangle(
                    frame,
                    tuple(current_points[0]),
                    tuple(current_points[1]),
                    YELLOW,
                    LINE_THICKNESS
                )

    # --------------------------------------------------------
    # Help bar
    # --------------------------------------------------------

    instructions = (
        f"MODE: {drawing_mode.upper()} | "
        "Left Click: point | "
        "ENTER: confirm | "
        "BACKSPACE: undo point | "
        "P: polygon | R: rectangle | "
        "U: undo region | C: clear | "
        "S: save | Q: skip video"
    )

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 35),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        instructions,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        WHITE,
        1,
        cv2.LINE_AA
    )

    # Resize only for the GUI.
    if display_scale != 1.0:
        frame = cv2.resize(
            frame,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_AREA
        )

    return frame


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):
    """Handle left-click annotation points."""

    global current_points

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    point = display_to_original(x, y)

    if drawing_mode == "polygon":
        current_points.append(point)

    elif drawing_mode == "rectangle":
        if len(current_points) < 2:
            current_points.append(point)

        else:
            print(
                "\nRectangle already has two corners."
                "\nPress ENTER to confirm it or BACKSPACE to edit it."
            )


# ============================================================
# CONFIRM CURRENT ROI
# ============================================================

def confirm_current_region():
    """Convert the current temporary shape into a confirmed ROI."""

    global current_points

    if drawing_mode == "polygon":
        if len(current_points) < 3:
            print("\nA polygon needs at least 3 points.")
            return

        final_points = [
            [int(x), int(y)]
            for x, y in current_points
        ]

    elif drawing_mode == "rectangle":
        if len(current_points) != 2:
            print(
                "\nA rectangle needs exactly 2 opposite corners."
            )
            return

        final_points = rectangle_points(
            current_points[0],
            current_points[1]
        )

    else:
        return

    suggested_name = f"ROI_{len(regions) + 1:02d}"

    print()

    name = input(
        f"Enter region/equipment name "
        f"[default: {suggested_name}]: "
    ).strip()

    if not name:
        name = suggested_name

    height, width = original_frame.shape[:2]

    normalized_points = [
        [x / width, y / height]
        for x, y in final_points
    ]

    region = {
        "id": len(regions) + 1,
        "name": name,
        "shape_type": drawing_mode,
        "pixel_points": final_points,
        "normalized_points": normalized_points,
    }

    regions.append(region)

    print(f"Confirmed region: {name}")
    print(f"Pixel coordinates: {final_points}")

    current_points = []


# ============================================================
# SAVE JSON + PKL + REFERENCE IMAGES
# ============================================================

def save_annotations(
    output_dir,
    video_path,
    reference_frame_number,
    fps,
    total_frames
):
    """
    Save:
      - regions.json
      - regions.pkl
      - reference_frame.png
      - annotated_reference_frame.png

    The full annotated video is produced separately afterwards.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    height, width = original_frame.shape[:2]

    annotation_data = {
        "video": {
            "filename": Path(video_path).name,
            "source_path": str(Path(video_path).resolve()),
            "width": width,
            "height": height,
            "fps": float(fps),
            "total_frames": int(total_frames),
            "reference_frame_number": int(
                reference_frame_number
            ),
            "reference_time_seconds": (
                float(reference_frame_number / fps)
                if fps > 0
                else None
            ),
        },
        "regions": regions,
    }

    # Save JSON.
    json_path = output_dir / "regions.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            annotation_data,
            file,
            indent=4
        )

    # Save PKL.
    pkl_path = output_dir / "regions.pkl"

    with open(
        pkl_path,
        "wb"
    ) as file:
        pickle.dump(
            annotation_data,
            file
        )

    # Save untouched reference frame.
    reference_path = (
        output_dir / "reference_frame.png"
    )

    cv2.imwrite(
        str(reference_path),
        original_frame
    )

    # Save annotated reference image.
    annotated = original_frame.copy()

    for region in regions:
        points = np.array(
            region["pixel_points"],
            dtype=np.int32
        )

        cv2.polylines(
            annotated,
            [points],
            True,
            RED,
            LINE_THICKNESS
        )

        min_x = int(np.min(points[:, 0]))
        min_y = int(np.min(points[:, 1]))

        cv2.putText(
            annotated,
            region["name"],
            (min_x, max(25, min_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            RED,
            2,
            cv2.LINE_AA
        )

    annotated_path = (
        output_dir / "annotated_reference_frame.png"
    )

    cv2.imwrite(
        str(annotated_path),
        annotated
    )

    print("\nAnnotation coordinates saved.")
    print(f"JSON : {json_path}")
    print(f"PKL  : {pkl_path}")
    print(f"Frame: {reference_path}")
    print(f"Image: {annotated_path}")


# ============================================================
# APPLY ROIs TO THE FULL VIDEO
# ============================================================

def create_annotated_video(
    video_path,
    output_path,
    regions_to_draw
):
    """
    Draw the same fixed ROIs on every frame and save a complete
    annotated MP4.

    Temperature extraction is NOT done here.
    """

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        capture.release()
        raise RuntimeError(
            f"Video reported invalid FPS ({fps}): {video_path}"
        )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        capture.release()

        raise RuntimeError(
            f"Could not create output video: {output_path}"
        )

    frame_number = 0

    print("\nApplying annotations to entire video...")

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            for region in regions_to_draw:
                points = np.array(
                    region["pixel_points"],
                    dtype=np.int32
                )

                cv2.polylines(
                    frame,
                    [points],
                    isClosed=True,
                    color=RED,
                    thickness=LINE_THICKNESS
                )

                min_x = int(
                    np.min(points[:, 0])
                )

                min_y = int(
                    np.min(points[:, 1])
                )

                cv2.putText(
                    frame,
                    region["name"],
                    (min_x, max(25, min_y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    RED,
                    2,
                    cv2.LINE_AA
                )

            writer.write(frame)

            frame_number += 1

            if frame_number % 100 == 0:
                percentage = (
                    100 * frame_number / total_frames
                    if total_frames > 0
                    else 0
                )

                print(
                    f"\rProcessed "
                    f"{frame_number}/{total_frames} "
                    f"frames ({percentage:.1f}%)",
                    end="",
                    flush=True
                )

    finally:
        capture.release()
        writer.release()

    print(
        f"\nAnnotated video finished:\n{output_path}"
    )


# ============================================================
# ANNOTATE ONE VIDEO
# ============================================================

def annotate_video(
    video_path,
    working_output_dir,
    frame_number=0
):
    """
    Open one video, let the user annotate ROIs, save the ROI
    files, and generate the full annotated video.

    Returns:
        True  -> completed successfully
        False -> user pressed Q and skipped/cancelled this video
    """

    global original_frame
    global display_scale
    global drawing_mode
    global current_points
    global regions

    video_path = Path(video_path)
    working_output_dir = Path(working_output_dir)

    # Reset every piece of state for this video.
    current_points = []
    regions = []
    drawing_mode = "polygon"
    original_frame = None
    display_scale = 1.0

    # Open video and extract the chosen reference frame.
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if total_frames <= 0:
        capture.release()
        raise RuntimeError(
            f"Video reported zero frames: {video_path}"
        )

    frame_number = max(
        0,
        min(frame_number, total_frames - 1)
    )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    success, frame = capture.read()

    capture.release()

    if not success:
        raise RuntimeError(
            f"Could not read frame {frame_number}"
        )

    original_frame = frame

    height, width = frame.shape[:2]

    display_scale = calculate_display_scale(
        width,
        height
    )

    print("\n============================================")
    print("THERMAL VIDEO ROI ANNOTATOR")
    print("============================================")
    print(f"Video: {video_path}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.3f}")
    print(f"Frames: {total_frames}")
    print(f"Reference frame: {frame_number}")

    if fps > 0:
        print(
            f"Reference time: "
            f"{frame_number / fps:.2f} seconds"
        )

    print("\nControls:")
    print("  Left Click  -> Add point")
    print("  ENTER       -> Confirm current ROI")
    print("  BACKSPACE   -> Remove previous point")
    print("  P           -> Polygon mode")
    print("  R           -> Rectangle mode")
    print("  U           -> Remove last confirmed ROI")
    print("  C           -> Clear current unfinished ROI")
    print("  S           -> Save this video and generate annotated MP4")
    print("  Q           -> Skip/cancel this video without replacing old output")
    print("  Ctrl+C      -> Stop the whole batch")

    # Create GUI window.
    window_name = (
        f"Thermal ROI Annotator - {video_path.name}"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_AUTOSIZE
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    user_saved = False

    while True:
        preview = draw_annotation_preview()

        cv2.imshow(
            window_name,
            preview
        )

        key = cv2.waitKey(20) & 0xFF

        # ENTER
        if key in (10, 13):
            confirm_current_region()

        # BACKSPACE
        elif key == 8:
            if current_points:
                removed = current_points.pop()
                print(f"Removed point: {removed}")

        # P -> polygon
        elif key in (ord("p"), ord("P")):
            current_points = []
            drawing_mode = "polygon"
            print("\nSwitched to POLYGON mode.")

        # R -> rectangle
        elif key in (ord("r"), ord("R")):
            current_points = []
            drawing_mode = "rectangle"
            print("\nSwitched to RECTANGLE mode.")

        # U -> undo confirmed region
        elif key in (ord("u"), ord("U")):
            if regions:
                removed_region = regions.pop()
                print(
                    f"\nRemoved region: "
                    f"{removed_region['name']}"
                )

        # C -> clear unfinished region
        elif key in (ord("c"), ord("C")):
            current_points = []
            print(
                "\nCurrent unfinished annotation cleared."
            )

        # S -> save current video
        elif key in (ord("s"), ord("S")):
            if not regions:
                print(
                    "\nNo regions have been annotated yet."
                )
                continue

            save_annotations(
                output_dir=working_output_dir,
                video_path=video_path,
                reference_frame_number=frame_number,
                fps=fps,
                total_frames=total_frames
            )

            user_saved = True
            break

        # Q -> cancel / skip this video
        elif key in (ord("q"), ord("Q")):
            print(
                "\nCurrent video cancelled. "
                "No completed output will be replaced."
            )
            break

    cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()

    if not user_saved:
        return False

    # Generate full annotated video.
    annotated_video_path = (
        working_output_dir / "annotated_video.mp4"
    )

    create_annotated_video(
        video_path=video_path,
        output_path=annotated_video_path,
        regions_to_draw=regions
    )

    return True


# ============================================================
# PROCESS ONE VIDEO SAFELY
# ============================================================

def process_one_video(
    video_path,
    final_output_dir,
    frame_number=0
):
    """
    Process one video in a temporary sibling folder.

    Only promote it to final_output_dir after ALL output files,
    including annotated_video.mp4, have completed.
    """

    final_output_dir = Path(final_output_dir)

    working_output_dir = final_output_dir.with_name(
        final_output_dir.name + ".__processing__"
    )

    # A temp folder means an older run was interrupted.
    if working_output_dir.exists():
        print(
            f"\nRemoving previous incomplete working folder:\n"
            f"{working_output_dir}"
        )
        safe_remove_directory(working_output_dir)

    working_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        completed = annotate_video(
            video_path=video_path,
            working_output_dir=working_output_dir,
            frame_number=frame_number
        )

        if not completed:
            safe_remove_directory(
                working_output_dir
            )
            return False

        if not is_annotation_base_complete(
            working_output_dir
        ):
            raise RuntimeError(
                "Annotation finished, but required base output files "
                "are missing or empty."
            )

        # Only write the provenance manifest after coordinates and the
        # full annotated MP4 have completed successfully.
        manifest = build_stage_01_manifest(
            video_path=video_path,
            working_output_dir=working_output_dir,
        )

        ps.save_json_atomic(
            working_output_dir / MANIFEST_FILENAME,
            manifest,
        )

        if not is_annotation_complete(working_output_dir):
            raise RuntimeError(
                "Step-01 manifest was written, but the tracked output "
                "still failed completion validation."
            )

        finalize_completed_output(
            working_dir=working_output_dir,
            final_dir=final_output_dir
        )

        print("\n============================================")
        print("VIDEO COMPLETED AND SAVED")
        print("============================================")
        print(final_output_dir)

        return True

    except KeyboardInterrupt:
        print(
            "\n\nBatch interrupted by user."
            "\nPreviously completed videos are safe."
            f"\nCurrent incomplete work is in:\n"
            f"{working_output_dir}"
        )
        raise

    except Exception:
        print(
            f"\nProcessing failed for:\n{video_path}"
            f"\nIncomplete working data kept at:\n"
            f"{working_output_dir}"
        )
        raise


# ============================================================
# BATCH DRIVER
# ============================================================

def process_input(
    input_path,
    output_root=DEFAULT_OUTPUT_ROOT,
    frame_number=0
):
    """
    Main batch workflow.

    For each discovered MP4:
      - determine its mirrored output directory
      - detect whether it is already completed
      - ask whether to process / redo / skip
      - save each finished video immediately before moving on
    """

    input_path = Path(input_path).resolve()
    output_root = Path(output_root).resolve()

    videos = discover_mp4_videos(
        input_path
    )

    if not videos:
        print(
            f"No MP4 videos found under:\n{input_path}"
        )
        return

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n============================================")
    print("THERMAL VIDEO BATCH ANNOTATOR")
    print("============================================")
    print(f"Input : {input_path}")
    print(f"Output: {output_root}")
    print(f"MP4 videos found: {len(videos)}")

    completed_this_run = 0
    skipped_this_run = 0
    failed_this_run = 0

    for index, video_path in enumerate(
        videos,
        start=1
    ):
        final_output_dir = build_output_dir(
            video_path=video_path,
            input_path=input_path,
            output_root=output_root
        )

        print("\n\n" + "=" * 70)
        print(
            f"[{index}/{len(videos)}] {video_path}"
        )
        print(
            f"Output folder: {final_output_dir}"
        )
        print("=" * 70)

        base_complete = is_annotation_base_complete(final_output_dir)
        tracked_complete = is_annotation_complete(final_output_dir)

        partial_output = (
            has_any_output(final_output_dir)
            and not base_complete
        )

        # ----------------------------------------------------
        # TRACKED COMPLETION: determine CURRENT vs STALE.
        # ----------------------------------------------------
        if tracked_complete:
            existing_manifest = ps.load_manifest(
                final_output_dir / MANIFEST_FILENAME
            )

            print("\nChecking Step-01 provenance...")
            current_source = ps.fingerprint_video(video_path)
            desired_reference = resolve_reference_frame(
                video_path,
                frame_number,
            )

            stale_reasons = step_01_stale_reasons(
                existing_manifest=existing_manifest or {},
                current_source=current_source,
                desired_reference_frame=desired_reference,
                output_dir=final_output_dir,
            )

            if not stale_reasons:
                print(
                    "\nSTEP 01 STATUS: CURRENT"
                    "\nThe existing annotations were produced from the "
                    "current raw video and requested reference frame."
                )

                should_process = prompt_yes_no(
                    "Redo this video anyway?",
                    default=False,
                )

            else:
                print("\nSTEP 01 STATUS: STALE")
                print("The existing output no longer matches the requested inputs:")

                for reason in stale_reasons:
                    print(f"  - {reason}")

                should_process = prompt_yes_no(
                    "Redo Step 01 using the current inputs?",
                    default=True,
                )

            if not should_process:
                print("Skipped.")
                skipped_this_run += 1
                continue

        # ----------------------------------------------------
        # LEGACY COMPLETION: valid old files, no manifest.
        # ----------------------------------------------------
        elif base_complete:
            print(
                "\nSTEP 01 STATUS: LEGACY / UNTRACKED"
                "\nFound regions.json + regions.pkl + annotated_video.mp4, "
                "but no stage_01_manifest.json."
                "\nThe old result is preserved, but its raw-video dependency "
                "cannot be proven automatically."
            )

            should_process = prompt_yes_no(
                "Redo this video to enable provenance tracking?",
                default=False,
            )

            if not should_process:
                print("Skipped. Existing legacy output was not changed.")
                skipped_this_run += 1
                continue

        # ----------------------------------------------------
        # PARTIAL / INCOMPLETE old result.
        # ----------------------------------------------------
        elif partial_output:
            print(
                "\nSTEP 01 STATUS: PARTIAL"
                "\nAn incomplete output folder exists. One or more base "
                "completion files are missing."
            )

            should_process = prompt_yes_no(
                f'Process "{video_path.name}" again from the beginning?',
                default=True,
            )

            if not should_process:
                print("Skipped.")
                skipped_this_run += 1
                continue

        # ----------------------------------------------------
        # NEW video.
        # ----------------------------------------------------
        else:
            print("\nSTEP 01 STATUS: NEW")

            should_process = prompt_yes_no(
                f'Processing "{video_path.name}"?',
                default=True,
            )

            if not should_process:
                print("Skipped.")
                skipped_this_run += 1
                continue

        # Run annotation.
        try:
            success = process_one_video(
                video_path=video_path,
                final_output_dir=final_output_dir,
                frame_number=frame_number
            )

            if success:
                completed_this_run += 1
            else:
                # Q inside GUI cancels current video and moves on.
                skipped_this_run += 1

        except KeyboardInterrupt:
            # Ctrl+C stops the entire batch.
            break

        except Exception as exc:
            failed_this_run += 1

            print(
                f"\nERROR: {exc}"
            )

            # One damaged video should not waste the rest.
            continue

    print("\n\n============================================")
    print("BATCH FINISHED")
    print("============================================")
    print(f"Completed this run: {completed_this_run}")
    print(f"Skipped:            {skipped_this_run}")
    print(f"Failed:             {failed_this_run}")
    print(f"Output root:        {output_root}")


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Annotate fixed ROIs in one thermal MP4 or "
            "recursively process every MP4 inside a folder."
        )
    )

    parser.add_argument(
        "input",
        help=(
            "Path to one .mp4 file OR a folder such as "
            "'raw-videos'. Folder mode searches recursively."
        )
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Stage-1 output root folder. "
            f"Default: {DEFAULT_OUTPUT_ROOT}"
        )
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help=(
            "Reference frame number used for annotation. "
            "In folder mode the same frame number is used "
            "for each video. Default: 0"
        )
    )

    args = parser.parse_args()

    process_input(
        input_path=args.input,
        output_root=args.output,
        frame_number=args.frame
    )
