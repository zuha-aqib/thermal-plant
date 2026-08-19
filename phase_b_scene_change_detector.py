import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import step_code as ps


# ============================================================
# PHASE B: AUTOMATIC CAMERA-STATE / SCENE-CHANGE DETECTION
# ============================================================
#
# This stage does NOT identify plants yet.
#
# Its job is only to answer:
#
#   1. Is the camera currently STABLE?
#   2. Is the camera currently MOVING?
#   3. Has it stopped at a NEW STABLE VIEW?
#   4. Has it returned to a stable view that we have seen before?
#
# Typical timeline:
#
#     VIEW_001
#     STABLE
#       |
#       v
#     MOVING
#       |
#       v
#     NEW_STABLE_VIEW -> VIEW_002
#       |
#       v
#     STABLE
#
# Later:
#
#     MOVING
#       |
#       v
#     VIEW_001 recognized again
#       |
#       v
#     STABLE
#
# No trained model is used here. This is classical computer vision.
# ============================================================


DEFAULT_OUTPUT_ROOT = "phase-b-scene-change-detection"

MANIFEST_FILENAME = "phase_b_manifest.json"

CORE_COMPLETION_FILES = (
    "scene_state_video.mp4",
    "scene_state_log.csv",
    "scene_events.csv",
    "view_library.json",
    "scene_summary.json",
)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)


# ============================================================
# PATH HELPERS
# ============================================================


def find_source_root_ancestor(video_path):
    """Find a known source root whose hierarchy should be preserved.

    Phase B is commonly run on raw videos or on tracked synthetic outputs.
    Preserving the pair/source parent prevents unrelated files such as
    ``incidental_001.mp4`` from colliding in the Phase-B output root.
    """
    video_path = Path(video_path).resolve()
    supported_roots = {
        "raw-videos",
        "synthetic-incidental-videos",
        "synthetic-moved-videos",
    }

    for parent in video_path.parents:
        if parent.name.lower() in supported_roots:
            return parent

    return None


def build_video_output_dir(video_path, output_root):
    """Mirror the hierarchy underneath known raw/synthetic roots.

    Examples:
        raw-videos/Furnace/Camera-01/video.mp4
            -> phase-b-scene-change-detection/Furnace/Camera-01/video/

        synthetic-incidental-videos/95__TO__97/incidental_001.mp4
            -> phase-b-scene-change-detection/95__TO__97/incidental_001/
    """
    video_path = Path(video_path).resolve()
    output_root = Path(output_root).resolve()

    source_root = find_source_root_ancestor(video_path)

    if source_root is not None:
        relative_video = video_path.relative_to(source_root)
        relative_parent = relative_video.parent
    else:
        relative_parent = Path()

    return output_root / relative_parent / video_path.stem


# ============================================================
# FRAME PREPARATION
# ============================================================


def resize_for_analysis(frame, analysis_width):
    """
    Resize only for motion/scene analysis.

    The output video remains at the ORIGINAL resolution.
    """
    height, width = frame.shape[:2]

    if width <= analysis_width:
        return frame.copy(), 1.0

    scale = analysis_width / float(width)

    resized = cv2.resize(
        frame,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale



def build_analysis_mask(
    frame,
    ignore_top_fraction,
    ignore_bottom_fraction,
    ignore_left_fraction,
    ignore_right_fraction,
    max_channel_difference,
):
    """
    Build a mask describing which parts of the image should be used
    for camera-motion and scene-comparison calculations.

    Two things are removed:

    1. Outer UI-heavy borders.
    2. Strongly colored overlays such as green labels/red markers.

    The thermal scene itself is mostly grayscale in the current
    footage, so colored screen overlays are poor tracking features.
    """
    height, width = frame.shape[:2]

    mask = np.ones(
        (height, width),
        dtype=np.uint8,
    ) * 255

    top = int(round(height * ignore_top_fraction))
    bottom = int(round(height * ignore_bottom_fraction))
    left = int(round(width * ignore_left_fraction))
    right = int(round(width * ignore_right_fraction))

    if top > 0:
        mask[:top, :] = 0

    if bottom > 0:
        mask[height - bottom :, :] = 0

    if left > 0:
        mask[:, :left] = 0

    if right > 0:
        mask[:, width - right :] = 0

    # Remove strongly colored screen overlays.
    frame_i16 = frame.astype(np.int16)
    channel_spread = (
        frame_i16.max(axis=2)
        - frame_i16.min(axis=2)
    )

    neutral = (
        channel_spread
        <= int(max_channel_difference)
    ).astype(np.uint8) * 255

    mask = cv2.bitwise_and(
        mask,
        neutral,
    )

    # Remove tiny isolated mask holes/noise.
    kernel = np.ones((3, 3), dtype=np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    return mask



def prepare_gray(frame):
    """
    Prepare a thermal image for structural comparison.

    CLAHE makes local edges easier to track even if the thermal camera
    changes its brightness/scale slightly.
    """
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    return gray



def build_edge_map(gray, mask):
    """
    Use edges as an extra motion signal.

    Edges are less sensitive than raw brightness to thermal auto-scale
    changes.
    """
    edges = cv2.Canny(
        gray,
        45,
        120,
    )

    edges = cv2.bitwise_and(
        edges,
        mask,
    )

    return edges


# ============================================================
# FRAME-TO-FRAME MOTION ESTIMATION
# ============================================================


def estimate_interframe_motion(
    previous_gray,
    current_gray,
    previous_mask,
    original_scale,
):
    """
    Estimate movement between two consecutive frames.

    This is deliberately simpler than Phase A's full recovery logic.
    Phase B mainly needs to know whether the camera is moving or still.

    Returns a dictionary containing:
        translation_pixels
        rotation_deg
        scale
        inliers
        inlier_ratio

    If tracking cannot be estimated, None is returned. A failed motion
    estimate combined with a large structural frame change is treated
    as evidence that the camera is moving.
    """
    corners = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=700,
        qualityLevel=0.01,
        minDistance=7,
        mask=previous_mask,
        blockSize=7,
    )

    if corners is None or len(corners) < 8:
        return None

    next_points, status, errors = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        corners,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            40,
            0.01,
        ),
    )

    if next_points is None or status is None:
        return None

    valid = status.reshape(-1).astype(bool)

    old_points = corners.reshape(-1, 2)[valid]
    new_points = next_points.reshape(-1, 2)[valid]

    if errors is not None:
        tracking_errors = errors.reshape(-1)[valid]
        error_ok = tracking_errors < 70.0
        old_points = old_points[error_ok]
        new_points = new_points[error_ok]

    if len(old_points) < 8:
        return None

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        old_points,
        new_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.995,
        refineIters=15,
    )

    if matrix is None:
        return None

    matched = len(old_points)

    inliers = (
        int(np.sum(inlier_mask))
        if inlier_mask is not None
        else 0
    )

    inlier_ratio = (
        inliers / matched
        if matched > 0
        else 0.0
    )

    # Movement of the CENTER is easier to interpret than raw affine tx/ty
    # if small rotation is also present.
    analysis_height, analysis_width = previous_gray.shape[:2]

    center = np.array(
        [
            analysis_width / 2.0,
            analysis_height / 2.0,
            1.0,
        ],
        dtype=np.float64,
    )

    transformed_center = matrix @ center

    dx_analysis = (
        transformed_center[0]
        - center[0]
    )

    dy_analysis = (
        transformed_center[1]
        - center[1]
    )

    translation_analysis = float(
        np.hypot(
            dx_analysis,
            dy_analysis,
        )
    )

    # Convert motion back to ORIGINAL-video pixels.
    if original_scale <= 0:
        translation_pixels = translation_analysis
    else:
        translation_pixels = (
            translation_analysis
            / original_scale
        )

    a = float(matrix[0, 0])
    c = float(matrix[1, 0])

    scale_value = float(
        np.sqrt(a * a + c * c)
    )

    rotation_deg = float(
        np.degrees(
            np.arctan2(c, a)
        )
    )

    return {
        "translation_pixels": translation_pixels,
        "rotation_deg": rotation_deg,
        "scale": scale_value,
        "matched_points": int(matched),
        "inliers": int(inliers),
        "inlier_ratio": float(inlier_ratio),
    }



def edge_change_ratio(previous_edges, current_edges, mask):
    """
    Measure how much the scene structure changed between frames.

    0.0 means almost identical edge structure.
    Larger values mean more structural change.

    This acts as a second signal if optical flow temporarily fails
    during a fast camera pan or motion blur.
    """
    valid = mask > 0

    if not np.any(valid):
        return 1.0

    # Expand edges slightly so a one-pixel shift does not count as a
    # completely different structure.
    kernel = np.ones((3, 3), dtype=np.uint8)

    previous_dilated = cv2.dilate(
        previous_edges,
        kernel,
        iterations=1,
    )

    current_dilated = cv2.dilate(
        current_edges,
        kernel,
        iterations=1,
    )

    difference = cv2.absdiff(
        previous_dilated,
        current_dilated,
    )

    changed = (
        (difference > 0)
        & valid
    )

    return float(
        np.mean(changed[valid])
    )


# ============================================================
# SCENE IDENTITY / VIEW RECOGNITION
# ============================================================


def scene_match(reference_gray, candidate_gray, reference_mask, candidate_mask):
    """
    Decide whether two STABLE frames show the same physical view.

    We use ORB features because the two stable frames may be separated
    by seconds/minutes and could have a noticeable camera offset.

    A same-view result should still contain many repeatable structural
    features whose positions agree under one affine transform.
    """
    orb = cv2.ORB_create(
        nfeatures=2200,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=21,
        patchSize=31,
        fastThreshold=8,
    )

    kp1, des1 = orb.detectAndCompute(
        reference_gray,
        reference_mask,
    )

    kp2, des2 = orb.detectAndCompute(
        candidate_gray,
        candidate_mask,
    )

    if (
        des1 is None
        or des2 is None
        or len(kp1) < 8
        or len(kp2) < 8
    ):
        return {
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "score": 0.0,
        }

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False,
    )

    knn = matcher.knnMatch(
        des1,
        des2,
        k=2,
    )

    good = []

    for pair in knn:
        if len(pair) != 2:
            continue

        best, second = pair

        if best.distance < 0.75 * second.distance:
            good.append(best)

    if len(good) < 6:
        return {
            "good_matches": len(good),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "score": 0.0,
        }

    points1 = np.float32([
        kp1[item.queryIdx].pt
        for item in good
    ])

    points2 = np.float32([
        kp2[item.trainIdx].pt
        for item in good
    ])

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        points1,
        points2,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=4000,
        confidence=0.995,
        refineIters=20,
    )

    if matrix is None or inlier_mask is None:
        return {
            "good_matches": len(good),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "score": 0.0,
        }

    inliers = int(np.sum(inlier_mask))
    ratio = inliers / len(good)

    # Score rewards BOTH quantity and agreement.
    # It is not a probability. It is only used to rank known views.
    score = float(inliers * ratio)

    return {
        "good_matches": int(len(good)),
        "inliers": int(inliers),
        "inlier_ratio": float(ratio),
        "score": score,
    }



def scene_match_is_acceptable(
    result,
    min_matches,
    min_inliers,
    min_inlier_ratio,
):
    """Apply conservative thresholds to a scene-match result."""
    return (
        result["good_matches"] >= int(min_matches)
        and result["inliers"] >= int(min_inliers)
        and result["inlier_ratio"] >= float(min_inlier_ratio)
    )


# ============================================================
# VIEW LIBRARY
# ============================================================


def add_new_view(
    view_library,
    frame_number,
    timestamp,
    original_frame,
    analysis_gray,
    analysis_mask,
    view_library_dir,
):
    """
    Create a new known stable view.

    The saved PNG becomes a human-readable scene library.
    """
    view_id = len(view_library) + 1
    view_name = f"VIEW_{view_id:03d}"

    image_path = (
        view_library_dir
        / f"{view_name}_reference.png"
    )

    cv2.imwrite(
        str(image_path),
        original_frame,
    )

    view = {
        "id": view_id,
        "name": view_name,
        "created_frame": int(frame_number),
        "created_timestamp_seconds": float(timestamp),
        "reference_image": str(image_path),
        "gray": analysis_gray.copy(),
        "mask": analysis_mask.copy(),
    }

    view_library.append(view)

    return view



def identify_stable_view(
    candidate_gray,
    candidate_mask,
    view_library,
    min_matches,
    min_inliers,
    min_inlier_ratio,
):
    """
    Compare a newly stable camera view against every previously seen
    view and return the best acceptable match.
    """
    best_view = None
    best_result = None

    for view in view_library:
        result = scene_match(
            view["gray"],
            candidate_gray,
            view["mask"],
            candidate_mask,
        )

        if not scene_match_is_acceptable(
            result,
            min_matches=min_matches,
            min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio,
        ):
            continue

        if (
            best_result is None
            or result["score"] > best_result["score"]
        ):
            best_view = view
            best_result = result

    return best_view, best_result


# ============================================================
# VIDEO OVERLAY
# ============================================================


def state_color(state):
    if state == "STABLE":
        return GREEN

    if state == "MOVING":
        return YELLOW

    if state == "NEW_STABLE_VIEW":
        return CYAN

    return WHITE



def draw_overlay(
    frame,
    state,
    view_name,
    motion_pixels,
    rotation_deg,
    edge_change,
    event_text,
):
    """Draw Phase-B telemetry on the output video."""
    lines = [
        f"State: {state}",
        f"View: {view_name}",
        (
            f"Frame motion: {motion_pixels:.2f}px | "
            f"rotation {rotation_deg:+.3f} deg"
        ),
        f"Structural change: {edge_change:.3f}",
    ]

    if event_text:
        lines.append(
            f"Event: {event_text}"
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 2

    widths = []
    heights = []

    for line in lines:
        (text_w, text_h), baseline = cv2.getTextSize(
            line,
            font,
            font_scale,
            thickness,
        )

        widths.append(text_w)
        heights.append(text_h + baseline)

    box_width = max(widths) + 30
    line_height = max(heights) + 10
    box_height = line_height * len(lines) + 14

    cv2.rectangle(
        frame,
        (10, 10),
        (10 + box_width, 10 + box_height),
        BLACK,
        -1,
    )

    color = state_color(state)

    y = 10 + line_height

    for index, line in enumerate(lines):
        line_color = (
            color
            if index == 0
            else WHITE
        )

        cv2.putText(
            frame,
            line,
            (22, y),
            font,
            font_scale,
            line_color,
            thickness,
            cv2.LINE_AA,
        )

        y += line_height


# ============================================================
# MAIN STATE MACHINE
# ============================================================


def process_video(video_path, output_dir, args):
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    view_library_dir = (
        output_dir / "view-library"
    )

    view_library_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(
            "Video reported invalid FPS/frame count."
        )

    output_video_path = (
        output_dir
        / "scene_state_video.mp4"
    )

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create output video: {output_video_path}"
        )

    log_csv_path = (
        output_dir
        / "scene_state_log.csv"
    )

    events_csv_path = (
        output_dir
        / "scene_events.csv"
    )

    log_file = open(
        log_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    )

    event_file = open(
        events_csv_path,
        "w",
        newline="",
        encoding="utf-8",
    )

    log_writer = csv.writer(log_file)
    event_writer = csv.writer(event_file)

    log_writer.writerow([
        "frame",
        "timestamp_seconds",
        "state",
        "view_id",
        "view_name",
        "motion_translation_pixels",
        "motion_rotation_deg",
        "motion_scale",
        "motion_inliers",
        "motion_inlier_ratio",
        "edge_change_ratio",
        "moving_evidence",
        "stable_evidence",
        "event",
    ])

    event_writer.writerow([
        "frame",
        "timestamp_seconds",
        "event",
        "from_state",
        "to_state",
        "previous_view",
        "current_view",
        "matched_known_view",
        "scene_good_matches",
        "scene_inliers",
        "scene_inlier_ratio",
        "scene_score",
    ])

    # --------------------------------------------------------
    # FIRST FRAME = first known stable view.
    # --------------------------------------------------------

    ok, first_frame = cap.read()

    if not ok:
        cap.release()
        writer.release()
        log_file.close()
        event_file.close()
        raise RuntimeError(
            "Could not read first frame."
        )

    first_analysis, analysis_scale = resize_for_analysis(
        first_frame,
        args.analysis_width,
    )

    first_mask = build_analysis_mask(
        first_analysis,
        ignore_top_fraction=args.ignore_top_fraction,
        ignore_bottom_fraction=args.ignore_bottom_fraction,
        ignore_left_fraction=args.ignore_left_fraction,
        ignore_right_fraction=args.ignore_right_fraction,
        max_channel_difference=args.max_channel_difference,
    )

    first_gray = prepare_gray(
        first_analysis
    )

    first_edges = build_edge_map(
        first_gray,
        first_mask,
    )

    view_library = []

    current_view = add_new_view(
        view_library=view_library,
        frame_number=0,
        timestamp=0.0,
        original_frame=first_frame,
        analysis_gray=first_gray,
        analysis_mask=first_mask,
        view_library_dir=view_library_dir,
    )

    current_state = "STABLE"

    moving_counter = 0
    stable_counter = 0

    # Show NEW_STABLE_VIEW visibly for a short period after an event.
    new_view_banner_remaining = 0

    previous_gray = first_gray
    previous_edges = first_edges
    previous_mask = first_mask

    state_counts = {
        "STABLE": 0,
        "MOVING": 0,
        "NEW_STABLE_VIEW": 0,
    }

    event_counts = {
        "MOVEMENT_STARTED": 0,
        "NEW_STABLE_VIEW": 0,
        "RETURN_TO_KNOWN_VIEW": 0,
        "SAME_VIEW_RESTABILIZED": 0,
    }

    print("\n============================================")
    print("PHASE B - SCENE CHANGE DETECTOR")
    print("============================================")
    print(f"Video      : {video_path}")
    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps:.3f}")
    print(f"Frames     : {total_frames}")
    print(f"Output     : {output_dir}")
    print(f"Initial    : {current_view['name']} / STABLE")

    # Write first frame.
    first_output = first_frame.copy()

    draw_overlay(
        first_output,
        state="STABLE",
        view_name=current_view["name"],
        motion_pixels=0.0,
        rotation_deg=0.0,
        edge_change=0.0,
        event_text="INITIAL_VIEW",
    )

    writer.write(first_output)

    state_counts["STABLE"] += 1

    log_writer.writerow([
        0,
        0.0,
        "STABLE",
        current_view["id"],
        current_view["name"],
        0.0,
        0.0,
        1.0,
        0,
        1.0,
        0.0,
        0,
        1,
        "INITIAL_VIEW",
    ])

    # --------------------------------------------------------
    # Remaining frames.
    # --------------------------------------------------------

    for frame_number in range(1, total_frames):
        ok, frame = cap.read()

        if not ok:
            break

        timestamp = frame_number / fps

        analysis_frame, current_scale = resize_for_analysis(
            frame,
            args.analysis_width,
        )

        current_mask = build_analysis_mask(
            analysis_frame,
            ignore_top_fraction=args.ignore_top_fraction,
            ignore_bottom_fraction=args.ignore_bottom_fraction,
            ignore_left_fraction=args.ignore_left_fraction,
            ignore_right_fraction=args.ignore_right_fraction,
            max_channel_difference=args.max_channel_difference,
        )

        current_gray = prepare_gray(
            analysis_frame
        )

        current_edges = build_edge_map(
            current_gray,
            current_mask,
        )

        # Use intersection of valid areas for frame-to-frame comparison.
        common_mask = cv2.bitwise_and(
            previous_mask,
            current_mask,
        )

        motion = estimate_interframe_motion(
            previous_gray=previous_gray,
            current_gray=current_gray,
            previous_mask=common_mask,
            original_scale=current_scale,
        )

        structural_change = edge_change_ratio(
            previous_edges=previous_edges,
            current_edges=current_edges,
            mask=common_mask,
        )

        if motion is None:
            motion_pixels = 0.0
            rotation_deg = 0.0
            motion_scale = 1.0
            motion_inliers = 0
            motion_inlier_ratio = 0.0

            # Failed tracking + obvious structural change is strong
            # evidence of camera motion.
            moving_evidence = (
                structural_change
                >= args.failed_tracking_change_threshold
            )

            stable_evidence = (
                structural_change
                <= args.stable_edge_change_threshold
            )

        else:
            motion_pixels = motion[
                "translation_pixels"
            ]

            rotation_deg = motion[
                "rotation_deg"
            ]

            motion_scale = motion[
                "scale"
            ]

            motion_inliers = motion[
                "inliers"
            ]

            motion_inlier_ratio = motion[
                "inlier_ratio"
            ]

            motion_quality_ok = (
                motion_inliers
                >= args.min_motion_inliers
                and motion_inlier_ratio
                >= args.min_motion_inlier_ratio
            )

            moving_evidence = (
                motion_quality_ok
                and (
                    motion_pixels
                    >= args.moving_translation_threshold
                    or abs(rotation_deg)
                    >= args.moving_rotation_threshold
                )
            ) or (
                structural_change
                >= args.moving_edge_change_threshold
            )

            stable_evidence = (
                motion_quality_ok
                and motion_pixels
                <= args.stable_translation_threshold
                and abs(rotation_deg)
                <= args.stable_rotation_threshold
                and structural_change
                <= args.stable_edge_change_threshold
            )

        event_text = ""

        # ====================================================
        # STATE MACHINE
        # ====================================================

        if current_state == "STABLE":
            if moving_evidence:
                moving_counter += 1
            else:
                moving_counter = 0

            if moving_counter >= args.moving_confirm_frames:
                old_state = current_state
                current_state = "MOVING"
                moving_counter = 0
                stable_counter = 0

                event_text = "MOVEMENT_STARTED"
                event_counts[event_text] += 1

                event_writer.writerow([
                    frame_number,
                    timestamp,
                    event_text,
                    old_state,
                    current_state,
                    current_view["name"],
                    current_view["name"],
                    "",
                    "",
                    "",
                    "",
                    "",
                ])

        elif current_state == "MOVING":
            if stable_evidence:
                stable_counter += 1
            else:
                stable_counter = 0

            # Camera has remained still for enough consecutive frames.
            if stable_counter >= args.stable_confirm_frames:
                old_state = current_state

                # Compare the NEW stable picture against all known views.
                matched_view, match_result = identify_stable_view(
                    candidate_gray=current_gray,
                    candidate_mask=current_mask,
                    view_library=view_library,
                    min_matches=args.scene_min_matches,
                    min_inliers=args.scene_min_inliers,
                    min_inlier_ratio=args.scene_min_inlier_ratio,
                )

                previous_view_name = current_view["name"]

                if matched_view is None:
                    # Truly unseen view.
                    current_view = add_new_view(
                        view_library=view_library,
                        frame_number=frame_number,
                        timestamp=timestamp,
                        original_frame=frame,
                        analysis_gray=current_gray,
                        analysis_mask=current_mask,
                        view_library_dir=view_library_dir,
                    )

                    current_state = "STABLE"
                    event_text = "NEW_STABLE_VIEW"
                    event_counts[event_text] += 1

                    new_view_banner_remaining = (
                        args.new_view_banner_frames
                    )

                    event_writer.writerow([
                        frame_number,
                        timestamp,
                        event_text,
                        old_state,
                        "STABLE",
                        previous_view_name,
                        current_view["name"],
                        "",
                        "",
                        "",
                        "",
                        "",
                    ])

                else:
                    # We have seen this view before.
                    current_view = matched_view
                    current_state = "STABLE"

                    if current_view["name"] == previous_view_name:
                        event_text = "SAME_VIEW_RESTABILIZED"
                    else:
                        event_text = "RETURN_TO_KNOWN_VIEW"

                    event_counts[event_text] += 1

                    event_writer.writerow([
                        frame_number,
                        timestamp,
                        event_text,
                        old_state,
                        "STABLE",
                        previous_view_name,
                        current_view["name"],
                        current_view["name"],
                        match_result["good_matches"],
                        match_result["inliers"],
                        match_result["inlier_ratio"],
                        match_result["score"],
                    ])

                stable_counter = 0
                moving_counter = 0

        # ----------------------------------------------------
        # Video display state.
        # NEW_STABLE_VIEW is shown as a temporary visual banner even
        # though internally the camera is already stable again.
        # ----------------------------------------------------

        display_state = current_state

        if new_view_banner_remaining > 0:
            display_state = "NEW_STABLE_VIEW"
            new_view_banner_remaining -= 1

        state_counts.setdefault(
            display_state,
            0,
        )

        state_counts[display_state] += 1

        output_frame = frame.copy()

        draw_overlay(
            output_frame,
            state=display_state,
            view_name=current_view["name"],
            motion_pixels=motion_pixels,
            rotation_deg=rotation_deg,
            edge_change=structural_change,
            event_text=event_text,
        )

        writer.write(output_frame)

        log_writer.writerow([
            frame_number,
            timestamp,
            display_state,
            current_view["id"],
            current_view["name"],
            motion_pixels,
            rotation_deg,
            motion_scale,
            motion_inliers,
            motion_inlier_ratio,
            structural_change,
            int(bool(moving_evidence)),
            int(bool(stable_evidence)),
            event_text,
        ])

        previous_gray = current_gray
        previous_edges = current_edges
        previous_mask = current_mask

        if (
            frame_number % 100 == 0
            or frame_number == total_frames - 1
        ):
            percentage = (
                100.0
                * (frame_number + 1)
                / total_frames
            )

            print(
                f"\rAnalyzing: "
                f"{frame_number + 1}/{total_frames} "
                f"({percentage:.1f}%)",
                end="",
                flush=True,
            )

    print()

    cap.release()
    writer.release()
    log_file.close()
    event_file.close()

    # --------------------------------------------------------
    # SAVE VIEW LIBRARY METADATA
    # --------------------------------------------------------

    view_metadata = []

    for view in view_library:
        view_metadata.append({
            "id": view["id"],
            "name": view["name"],
            "created_frame": view["created_frame"],
            "created_timestamp_seconds": (
                view["created_timestamp_seconds"]
            ),
            "reference_image": view["reference_image"],
        })

    views_json_path = (
        output_dir / "view_library.json"
    )

    with open(
        views_json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "video": video_path.name,
                "views": view_metadata,
            },
            file,
            indent=4,
        )

    summary_path = (
        output_dir / "scene_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "video": video_path.name,
                "fps": fps,
                "total_frames": total_frames,
                "known_views": len(view_library),
                "state_frame_counts": state_counts,
                "event_counts": event_counts,
                "output_video": str(output_video_path),
                "scene_state_log": str(log_csv_path),
                "scene_events": str(events_csv_path),
                "view_library": str(views_json_path),
            },
            file,
            indent=4,
        )

    print("\n============================================")
    print("PHASE B COMPLETE")
    print("============================================")
    print(f"Known stable views : {len(view_library)}")
    print(f"State video        : {output_video_path}")
    print(f"Frame log          : {log_csv_path}")
    print(f"Events             : {events_csv_path}")
    print(f"View library       : {views_json_path}")
    print(f"Summary            : {summary_path}")




# ============================================================
# PHASE-B STATE / PROVENANCE HELPERS
# ============================================================


def normalized_phase_b_settings(args):
    """Return every CLI setting that can affect Phase-B output."""
    return {
        "analysis_width": int(args.analysis_width),
        "ignore_top_fraction": float(args.ignore_top_fraction),
        "ignore_bottom_fraction": float(args.ignore_bottom_fraction),
        "ignore_left_fraction": float(args.ignore_left_fraction),
        "ignore_right_fraction": float(args.ignore_right_fraction),
        "max_channel_difference": int(args.max_channel_difference),
        "moving_translation_threshold_pixels": float(args.moving_translation_threshold),
        "stable_translation_threshold_pixels": float(args.stable_translation_threshold),
        "moving_rotation_threshold_deg": float(args.moving_rotation_threshold),
        "stable_rotation_threshold_deg": float(args.stable_rotation_threshold),
        "moving_edge_change_threshold": float(args.moving_edge_change_threshold),
        "stable_edge_change_threshold": float(args.stable_edge_change_threshold),
        "failed_tracking_change_threshold": float(args.failed_tracking_change_threshold),
        "min_motion_inliers": int(args.min_motion_inliers),
        "min_motion_inlier_ratio": float(args.min_motion_inlier_ratio),
        "moving_confirm_frames": int(args.moving_confirm_frames),
        "stable_confirm_frames": int(args.stable_confirm_frames),
        "scene_min_matches": int(args.scene_min_matches),
        "scene_min_inliers": int(args.scene_min_inliers),
        "scene_min_inlier_ratio": float(args.scene_min_inlier_ratio),
        "new_view_banner_frames": int(args.new_view_banner_frames),
    }


def phase_b_view_library_complete(output_dir):
    """A completed Phase-B run should contain at least VIEW_001 reference."""
    view_dir = Path(output_dir) / "view-library"

    if not view_dir.is_dir():
        return False

    return any(
        path.is_file() and path.stat().st_size > 0
        for path in view_dir.glob("VIEW_*_reference.png")
    )


def is_phase_b_legacy_complete(output_dir):
    """Recognize a complete pre-manifest Phase-B result."""
    return (
        ps.verify_required_files(output_dir, CORE_COMPLETION_FILES)
        and phase_b_view_library_complete(output_dir)
    )


def is_phase_b_tracked_complete(output_dir):
    """True only for complete output + readable Phase-B manifest."""
    output_dir = Path(output_dir)

    if not ps.load_manifest(output_dir / MANIFEST_FILENAME):
        return False

    return is_phase_b_legacy_complete(output_dir)


def build_phase_b_manifest(*, source_fingerprint, settings):
    """Create the provenance receipt for one completed Phase-B run."""
    return ps.make_manifest(
        "phase_b_scene_change_detector",
        source_video=source_fingerprint,
        settings=settings,
        outputs={
            "scene_state_video": "scene_state_video.mp4",
            "scene_state_log_csv": "scene_state_log.csv",
            "scene_events_csv": "scene_events.csv",
            "view_library_json": "view_library.json",
            "scene_summary_json": "scene_summary.json",
            "view_library_directory": "view-library",
        },
    )


def phase_b_stale_reasons(
    *,
    existing_manifest,
    current_source,
    requested_settings,
):
    """Explain exactly why an existing tracked Phase-B result is stale."""
    reasons = []

    if not ps.same_file_content(
        existing_manifest.get("source_video", {}),
        current_source,
    ):
        reasons.append("Input video content changed")

    old_settings = existing_manifest.get("settings", {})

    friendly_names = {
        "analysis_width": "analysis width",
        "ignore_top_fraction": "ignored top fraction",
        "ignore_bottom_fraction": "ignored bottom fraction",
        "ignore_left_fraction": "ignored left fraction",
        "ignore_right_fraction": "ignored right fraction",
        "max_channel_difference": "UI color-rejection threshold",
        "moving_translation_threshold_pixels": "moving translation threshold",
        "stable_translation_threshold_pixels": "stable translation threshold",
        "moving_rotation_threshold_deg": "moving rotation threshold",
        "stable_rotation_threshold_deg": "stable rotation threshold",
        "moving_edge_change_threshold": "moving edge-change threshold",
        "stable_edge_change_threshold": "stable edge-change threshold",
        "failed_tracking_change_threshold": "failed-tracking structural-change threshold",
        "min_motion_inliers": "minimum motion inliers",
        "min_motion_inlier_ratio": "minimum motion inlier ratio",
        "moving_confirm_frames": "moving confirmation frame count",
        "stable_confirm_frames": "stable confirmation frame count",
        "scene_min_matches": "same-view minimum ORB matches",
        "scene_min_inliers": "same-view minimum RANSAC inliers",
        "scene_min_inlier_ratio": "same-view minimum inlier ratio",
        "new_view_banner_frames": "new-view banner duration",
    }

    for key, new_value in requested_settings.items():
        old_value = old_settings.get(key)

        if old_value != new_value:
            reasons.append(
                f"Phase B {friendly_names.get(key, key)} changed "
                f"({old_value} -> {new_value})"
            )

    return reasons


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase B: automatically classify the camera as STABLE or "
            "MOVING and detect/remember new stable camera views."
        )
    )

    parser.add_argument(
        "video",
        help="Path to the camera/thermal MP4",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Output root. The raw-videos hierarchy is mirrored. "
            f"Default: {DEFAULT_OUTPUT_ROOT}"
        ),
    )

    parser.add_argument(
        "--analysis-width",
        type=int,
        default=640,
        help=(
            "Width used internally for motion analysis. Smaller is faster. "
            "The saved video stays full resolution. Default: 640"
        ),
    )

    # --------------------------------------------------------
    # Ignore common thermal-camera UI zones.
    # --------------------------------------------------------

    parser.add_argument(
        "--ignore-top-fraction",
        type=float,
        default=0.05,
        help="Ignore this fraction of the top of the image. Default 0.05.",
    )

    parser.add_argument(
        "--ignore-bottom-fraction",
        type=float,
        default=0.03,
        help="Ignore this fraction of the bottom. Default 0.03.",
    )

    parser.add_argument(
        "--ignore-left-fraction",
        type=float,
        default=0.02,
        help="Ignore this fraction of the left edge. Default 0.02.",
    )

    parser.add_argument(
        "--ignore-right-fraction",
        type=float,
        default=0.12,
        help=(
            "Ignore this fraction of the right edge, where the thermal "
            "scale/UI commonly lives. Default 0.12."
        ),
    )

    parser.add_argument(
        "--max-channel-difference",
        type=int,
        default=25,
        help=(
            "Pixels with stronger color than this are ignored as possible "
            "camera UI overlays. Default: 25."
        ),
    )

    # --------------------------------------------------------
    # STABLE vs MOVING thresholds.
    # --------------------------------------------------------

    parser.add_argument(
        "--moving-translation-threshold",
        type=float,
        default=2.0,
        help=(
            "Frame-to-frame center movement in ORIGINAL pixels that counts "
            "as camera motion. Default: 2.0 px."
        ),
    )

    parser.add_argument(
        "--stable-translation-threshold",
        type=float,
        default=0.8,
        help=(
            "Frame-to-frame movement below this can count as stable. "
            "Default: 0.8 px."
        ),
    )

    parser.add_argument(
        "--moving-rotation-threshold",
        type=float,
        default=0.12,
        help="Frame rotation that counts as moving. Default: 0.12 degrees.",
    )

    parser.add_argument(
        "--stable-rotation-threshold",
        type=float,
        default=0.06,
        help="Rotation below this can count as stable. Default: 0.06 degrees.",
    )

    parser.add_argument(
        "--moving-edge-change-threshold",
        type=float,
        default=0.12,
        help=(
            "Structural edge-change ratio that alone indicates movement. "
            "Default: 0.12."
        ),
    )

    parser.add_argument(
        "--stable-edge-change-threshold",
        type=float,
        default=0.07,
        help=(
            "Structural edge-change ratio below this can count as stable. "
            "Default: 0.07."
        ),
    )

    parser.add_argument(
        "--failed-tracking-change-threshold",
        type=float,
        default=0.10,
        help=(
            "If optical flow fails but structural change exceeds this, "
            "treat the frame as moving. Default: 0.10."
        ),
    )

    parser.add_argument(
        "--min-motion-inliers",
        type=int,
        default=10,
        help="Minimum RANSAC motion inliers. Default: 10.",
    )

    parser.add_argument(
        "--min-motion-inlier-ratio",
        type=float,
        default=0.35,
        help="Minimum motion inlier ratio. Default: 0.35.",
    )

    parser.add_argument(
        "--moving-confirm-frames",
        type=int,
        default=3,
        help=(
            "Consecutive moving frames before STABLE -> MOVING. Default: 3."
        ),
    )

    parser.add_argument(
        "--stable-confirm-frames",
        type=int,
        default=12,
        help=(
            "Consecutive stable frames before MOVING is considered settled. "
            "Default: 12."
        ),
    )

    # --------------------------------------------------------
    # Stable scene/view matching.
    # --------------------------------------------------------

    parser.add_argument(
        "--scene-min-matches",
        type=int,
        default=20,
        help="Minimum ORB good matches for same-view recognition. Default 20.",
    )

    parser.add_argument(
        "--scene-min-inliers",
        type=int,
        default=12,
        help="Minimum scene-match RANSAC inliers. Default 12.",
    )

    parser.add_argument(
        "--scene-min-inlier-ratio",
        type=float,
        default=0.30,
        help="Minimum scene-match inlier ratio. Default 0.30.",
    )

    parser.add_argument(
        "--new-view-banner-frames",
        type=int,
        default=30,
        help=(
            "How long NEW_STABLE_VIEW remains visibly shown in the output "
            "video after a new view is detected. Default: 30 frames."
        ),
    )

    args = parser.parse_args()

    video_path = Path(args.video).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    output_dir = build_video_output_dir(
        video_path=video_path,
        output_root=args.output,
    )

    requested_settings = normalized_phase_b_settings(args)

    # Validate the input before any existing completed output can be touched.
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or total_frames <= 0:
        raise RuntimeError("Video reported invalid FPS/frame count.")

    print("\n============================================")
    print("PHASE B - SCENE CHANGE DETECTOR")
    print("============================================")
    print(f"Video      : {video_path}")
    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps:.3f}")
    print(f"Frames     : {total_frames}")
    print(f"Output     : {output_dir}")

    # --------------------------------------------------------
    # CURRENT / STALE / LEGACY / PARTIAL / NEW classification.
    # --------------------------------------------------------
    print("\nChecking Phase-B dependencies...")
    current_source = ps.fingerprint_video(video_path)

    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_file_present = ps.file_exists_and_nonempty(manifest_path)
    tracked_complete = is_phase_b_tracked_complete(output_dir)
    legacy_complete = (
        is_phase_b_legacy_complete(output_dir)
        and not manifest_file_present
    )
    partial_output = (
        ps.has_any_output(output_dir)
        and not tracked_complete
        and not legacy_complete
    )

    should_process = True

    if tracked_complete:
        existing_manifest = ps.load_manifest(manifest_path) or {}
        stale_reasons = phase_b_stale_reasons(
            existing_manifest=existing_manifest,
            current_source=current_source,
            requested_settings=requested_settings,
        )

        if not stale_reasons:
            print("\nPHASE B STATUS: CURRENT")
            print("Existing output was generated using:")
            print("  - Current input video")
            print("  - Same Phase B scene/motion settings")
            should_process = ps.prompt_yes_no(
                "Re-run Phase B anyway?",
                default=False,
            )
        else:
            print("\nPHASE B STATUS: STALE")
            print("The existing Phase-B output was made from older inputs/settings:")
            for reason in stale_reasons:
                print(f"  - {reason}")
            should_process = ps.prompt_yes_no(
                "Re-run Phase B using the current inputs/settings?",
                default=True,
            )

    elif legacy_complete:
        print("\nPHASE B STATUS: LEGACY / UNTRACKED")
        print(
            "A complete older Phase-B result exists, but there is no "
            "phase_b_manifest.json proving which video/settings created it."
        )
        should_process = ps.prompt_yes_no(
            "Re-run once to enable dependency tracking?",
            default=False,
        )

    elif partial_output:
        print("\nPHASE B STATUS: PARTIAL")
        print("An incomplete Phase-B output folder exists.")
        should_process = ps.prompt_yes_no(
            "Restart Phase B from the beginning?",
            default=True,
        )

    else:
        print("\nPHASE B STATUS: NEW")
        should_process = ps.prompt_yes_no(
            f'Run Phase B for "{video_path.name}"?',
            default=True,
        )

    if not should_process:
        print("Skipped. Existing output was not changed.")
        return

    # --------------------------------------------------------
    # Safe rerun: produce everything in a sibling temp folder first.
    # --------------------------------------------------------
    working_dir = output_dir.with_name(output_dir.name + ".__processing__")

    if working_dir.exists():
        print(f"\nRemoving previous incomplete working folder:\n{working_dir}")
        ps.safe_remove(working_dir)

    working_dir.mkdir(parents=True, exist_ok=True)

    try:
        process_video(
            video_path=video_path,
            output_dir=working_dir,
            args=args,
        )

        if not is_phase_b_legacy_complete(working_dir):
            raise RuntimeError(
                "Phase B finished but one or more required outputs "
                "or view-library reference images are missing/empty."
            )

        manifest = build_phase_b_manifest(
            source_fingerprint=current_source,
            settings=requested_settings,
        )
        ps.save_json_atomic(
            working_dir / MANIFEST_FILENAME,
            manifest,
        )

        if not is_phase_b_tracked_complete(working_dir):
            raise RuntimeError("Tracked Phase-B completion validation failed.")

        ps.promote_completed_directory(
            working_dir=working_dir,
            final_dir=output_dir,
        )

        print("\n============================================")
        print("PHASE B COMPLETED AND SAVED")
        print("============================================")
        print(f"Output   : {output_dir}")
        print(f"Manifest : {output_dir / MANIFEST_FILENAME}")

    except KeyboardInterrupt:
        print(
            "\n\nPhase B interrupted. The previous completed output, if any, "
            "was not replaced."
            f"\nIncomplete new work is kept at:\n{working_dir}"
        )
        raise

    except Exception:
        print(
            "\nPhase B failed. The previous completed output, if any, "
            "was not replaced."
            f"\nIncomplete new work is kept at:\n{working_dir}"
        )
        raise


if __name__ == "__main__":
    main()
