import argparse
import csv
import json
import math
import pickle
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PHASE A: ACCIDENTAL CAMERA MOVEMENT COMPENSATION
# ============================================================
#
# This script solves the "small accidental movement" problem:
#   - wind moves the camera slightly
#   - a bird / pole / vibration nudges the camera
#   - the monitored equipment is still the SAME equipment
#   - therefore the saved ROI polygons should move with the scene
#
# It does NOT calculate temperature yet. We first validate motion
# tracking in isolation so we do not disturb the working static
# thermal pipeline.
#
# Core strategy:
#   1. Keep a trusted reference / last-good frame.
#   2. Track plant features into the current frame.
#   3. Estimate a global limited affine transform using RANSAC.
#   4. Apply that transform to the ORIGINAL polygon coordinates.
#   5. If optical flow fails, retry using ORB feature matching.
#   6. If both fail, hold the last good polygons and try recovery
#      again from the last trusted frame on the next frame.
#
# A limited affine transform handles the accidental-movement case:
#   - x/y translation
#   - rotation
#   - small uniform scaling
# ============================================================


# ----------------------------
# Visual settings
# ----------------------------
RED = (0, 0, 255)
YELLOW = (0, 255, 255)   # optional old/fixed polygons for debugging
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LINE_THICKNESS = 2

DEFAULT_OUTPUT_ROOT = "phase-a-accidental-camera-motion"


# ============================================================
# FILE / PATH HELPERS
# ============================================================

def load_json(path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_raw_videos_ancestor(video_path):
    """Find the nearest parent folder literally named raw-videos."""
    video_path = Path(video_path).resolve()

    for parent in video_path.parents:
        if parent.name.lower() == "raw-videos":
            return parent

    return None


def build_video_output_dir(video_path, output_root):
    """
    Preserve nested folders underneath raw-videos.

    Example:
        raw-videos/Furnace/Camera-1/sample.mp4

    becomes:
        phase-a-accidental-camera-motion/Furnace/Camera-1/sample/
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
# IMAGE PREPROCESSING
# ============================================================

def prepare_tracking_gray(frame):
    """
    Convert a thermal frame to a contrast-enhanced grayscale image.

    Thermal cameras can auto-rescale brightness over time. CLAHE
    improves local contrast, so edges/corners remain easier to track
    even when global grayscale intensity changes slightly.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(gray)


# ============================================================
# AFFINE / POINT HELPERS
# ============================================================

def identity_h():
    """3x3 identity transform."""
    return np.eye(3, dtype=np.float64)


def affine_2x3_to_3x3(matrix):
    """Convert OpenCV 2x3 affine matrix into 3x3 homogeneous form."""
    h = identity_h()
    h[:2, :] = matrix
    return h


def transform_points(points, h):
    """Apply a 3x3 transform to [[x,y], ...] polygon points."""
    pts = np.asarray(points, dtype=np.float64)

    homogeneous = np.hstack([
        pts,
        np.ones((len(pts), 1), dtype=np.float64),
    ])

    moved = (h @ homogeneous.T).T
    moved_xy = moved[:, :2] / moved[:, 2:3]

    return moved_xy


def affine_metrics(h, width, height):
    """
    Convert an affine transform into understandable camera movement.

    dx/dy are defined as movement of the IMAGE CENTER, which is easier
    to interpret than raw affine tx/ty when rotation is also present.
    """
    a = float(h[0, 0])
    c = float(h[1, 0])

    scale = math.sqrt(a * a + c * c)
    rotation_deg = math.degrees(math.atan2(c, a))

    center = np.array([
        width / 2.0,
        height / 2.0,
        1.0,
    ], dtype=np.float64)

    moved_center = h @ center
    moved_center = moved_center[:2] / moved_center[2]

    dx = float(moved_center[0] - center[0])
    dy = float(moved_center[1] - center[1])

    return {
        "dx": dx,
        "dy": dy,
        "translation": math.hypot(dx, dy),
        "rotation_deg": rotation_deg,
        "scale": scale,
    }


# ============================================================
# TRACKING MASK
# ============================================================

def build_tracking_mask(frame_shape, regions, reference_to_anchor, margin_pixels=80):
    """
    Tell the feature tracker where to look.

    We intentionally prioritize the annotated equipment and nearby
    structure instead of the whole screen.

    Why this matters:
      The thermal camera adds overlays such as temperature text,
      scale bars, timestamps and crosshairs. Those UI elements may
      remain fixed in screen coordinates even when the physical scene
      moves. We do NOT want the camera-motion tracker to lock onto UI.
    """
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for region in regions:
        moved = transform_points(
            region["pixel_points"],
            reference_to_anchor,
        )

        polygon = np.rint(moved).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255)

    # Include edges/structures close to the equipment too.
    if margin_pixels > 0:
        k = 2 * int(margin_pixels) + 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k),
        )

        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# ============================================================
# MOTION ESTIMATION METHOD 1: OPTICAL FLOW
# ============================================================

def estimate_with_optical_flow(anchor_gray, current_gray, tracking_mask, max_corners, min_corner_distance):
    """
    Fast primary method.

    1. Detect Shi-Tomasi corners in the last trusted frame.
    2. Track them into the current frame using pyramidal LK flow.
    3. Estimate a limited affine transform with RANSAC.
    """
    corners = cv2.goodFeaturesToTrack(
        anchor_gray,
        maxCorners=int(max_corners),
        qualityLevel=0.01,
        minDistance=float(min_corner_distance),
        mask=tracking_mask,
        blockSize=7,
        useHarrisDetector=False,
    )

    if corners is None or len(corners) < 6:
        return None

    next_points, status, errors = cv2.calcOpticalFlowPyrLK(
        anchor_gray,
        current_gray,
        corners,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            40,
            0.01,
        ),
    )

    if next_points is None or status is None:
        return None

    status = status.reshape(-1).astype(bool)

    old_points = corners.reshape(-1, 2)[status]
    new_points = next_points.reshape(-1, 2)[status]

    # Remove extremely poor LK tracks.
    if errors is not None and len(old_points) > 0:
        good_errors = errors.reshape(-1)[status]
        error_mask = good_errors < 60.0

        old_points = old_points[error_mask]
        new_points = new_points[error_mask]

    if len(old_points) < 6:
        return None

    matrix, inliers = cv2.estimateAffinePartial2D(
        old_points,
        new_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2500,
        confidence=0.995,
        refineIters=15,
    )

    if matrix is None:
        return None

    inlier_count = int(np.sum(inliers)) if inliers is not None else 0

    return {
        "method": "LK_OPTICAL_FLOW",
        "matrix": matrix.astype(np.float64),
        "matched_points": int(len(old_points)),
        "inlier_count": inlier_count,
    }


# ============================================================
# MOTION ESTIMATION METHOD 2: ORB FALLBACK
# ============================================================

def estimate_with_orb(anchor_gray, current_gray, tracking_mask):
    """
    Recovery method used only when optical flow fails validation.

    ORB can recover from a somewhat larger sudden shift because it
    matches feature descriptors rather than assuming every feature
    stayed close to its previous location.
    """
    orb = cv2.ORB_create(
        nfeatures=1800,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=21,
        patchSize=31,
        fastThreshold=10,
    )

    kp1, des1 = orb.detectAndCompute(anchor_gray, tracking_mask)
    kp2, des2 = orb.detectAndCompute(current_gray, None)

    if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(des1, des2, k=2)

    good = []

    for pair in pairs:
        if len(pair) != 2:
            continue

        first, second = pair

        if first.distance < 0.75 * second.distance:
            good.append(first)

    if len(good) < 6:
        return None

    old_points = np.float32([kp1[m.queryIdx].pt for m in good])
    new_points = np.float32([kp2[m.trainIdx].pt for m in good])

    matrix, inliers = cv2.estimateAffinePartial2D(
        old_points,
        new_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=4000,
        confidence=0.995,
        refineIters=20,
    )

    if matrix is None:
        return None

    inlier_count = int(np.sum(inliers)) if inliers is not None else 0

    return {
        "method": "ORB_FALLBACK",
        "matrix": matrix.astype(np.float64),
        "matched_points": int(len(good)),
        "inlier_count": inlier_count,
    }


# ============================================================
# TRANSFORM VALIDATION
# ============================================================

def validate_candidate(candidate, reference_to_anchor, width, height, args):
    """
    Protect the polygons from bad feature matches.

    We validate:
      - how many matches support the transform
      - RANSAC inlier ratio
      - per-update translation/rotation/scale
      - total movement relative to the original reference

    Large / implausible movement is NOT blindly accepted because it
    may actually mean the camera intentionally switched to a new view,
    which belongs to the later incidental-movement phase.
    """
    if candidate is None:
        return False, "NO_TRANSFORM", None

    matched = int(candidate["matched_points"])
    inliers = int(candidate["inlier_count"])

    if matched <= 0:
        return False, "NO_MATCHES", None

    inlier_ratio = inliers / matched

    if inliers < args.min_inliers:
        return False, "TOO_FEW_INLIERS", None

    if inlier_ratio < args.min_inlier_ratio:
        return False, "LOW_INLIER_RATIO", None

    delta_h = affine_2x3_to_3x3(candidate["matrix"])
    delta_metrics = affine_metrics(delta_h, width, height)

    if delta_metrics["translation"] > args.max_delta_translation:
        return False, "DELTA_TRANSLATION_TOO_LARGE", None

    if abs(delta_metrics["rotation_deg"]) > args.max_delta_rotation:
        return False, "DELTA_ROTATION_TOO_LARGE", None

    if not (args.min_delta_scale <= delta_metrics["scale"] <= args.max_delta_scale):
        return False, "DELTA_SCALE_IMPLAUSIBLE", None

    # Compose reference->anchor and anchor->current.
    reference_to_current = delta_h @ reference_to_anchor
    cumulative_metrics = affine_metrics(reference_to_current, width, height)

    if cumulative_metrics["translation"] > args.max_cumulative_translation:
        return False, "CUMULATIVE_TRANSLATION_TOO_LARGE", None

    if abs(cumulative_metrics["rotation_deg"]) > args.max_cumulative_rotation:
        return False, "CUMULATIVE_ROTATION_TOO_LARGE", None

    if not (args.min_cumulative_scale <= cumulative_metrics["scale"] <= args.max_cumulative_scale):
        return False, "CUMULATIVE_SCALE_IMPLAUSIBLE", None

    result = {
        **candidate,
        "inlier_ratio": float(inlier_ratio),
        "delta_h": delta_h,
        "delta_metrics": delta_metrics,
        "reference_to_current": reference_to_current,
        "cumulative_metrics": cumulative_metrics,
    }

    return True, "OK", result


def estimate_current_transform(anchor_gray, current_gray, tracking_mask, reference_to_anchor, width, height, args):
    """
    Primary = optical flow.
    Recovery = ORB on the same frame.
    """
    lk = estimate_with_optical_flow(
        anchor_gray,
        current_gray,
        tracking_mask,
        max_corners=args.max_corners,
        min_corner_distance=args.min_corner_distance,
    )

    valid, lk_reason, result = validate_candidate(
        lk,
        reference_to_anchor,
        width,
        height,
        args,
    )

    if valid:
        return result, "LK_ACCEPTED"

    orb = estimate_with_orb(
        anchor_gray,
        current_gray,
        tracking_mask,
    )

    valid, orb_reason, result = validate_candidate(
        orb,
        reference_to_anchor,
        width,
        height,
        args,
    )

    if valid:
        return result, f"ORB_RECOVERY_AFTER_{lk_reason}"

    return None, f"TRACKING_FAILED:LK={lk_reason};ORB={orb_reason}"


# ============================================================
# PASS 1: COMPUTE REFERENCE -> CURRENT TRANSFORM FOR EACH FRAME
# ============================================================

def compute_transforms(video_path, regions_data, args):
    """
    Real-time style forward tracking.

    IMPORTANT RECOVERY BEHAVIOR:
      If a frame is blurred during the camera bump and cannot be
      aligned, it does NOT become the next anchor.

      The next frame is again compared with the LAST TRUSTED frame.
      This allows the system to recover the full displacement after
      one or two bad movement/blur frames.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    reference_frame_number = int(
        regions_data.get("video", {}).get("reference_frame_number", 0)
    )

    reference_frame_number = max(0, min(reference_frame_number, total_frames - 1))

    # Phase A is intended to operate forward in real time from a stable
    # reference. Frames before the stored reference are left unchanged.
    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_number)
    ok, reference_frame = cap.read()

    if not ok:
        cap.release()
        raise RuntimeError(f"Could not read reference frame {reference_frame_number}")

    reference_gray = prepare_tracking_gray(reference_frame)

    transforms = {}
    logs = {}

    # Frames before reference cannot be estimated causally from that
    # future reference, so keep identity. Normally reference is frame 0.
    for frame_number in range(reference_frame_number):
        transforms[frame_number] = identity_h()
        logs[frame_number] = {
            "status": "BEFORE_REFERENCE",
            "method": "NONE",
            "matched_points": 0,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "reason": "BEFORE_REFERENCE",
        }

    transforms[reference_frame_number] = identity_h()
    logs[reference_frame_number] = {
        "status": "REFERENCE",
        "method": "REFERENCE",
        "matched_points": 0,
        "inlier_count": 0,
        "inlier_ratio": 1.0,
        "reason": "REFERENCE",
    }

    regions = regions_data["regions"]

    anchor_gray = reference_gray
    reference_to_anchor = identity_h()
    failed_since_anchor = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_number + 1)

    print("\nPASS 1: Estimating accidental camera movement...")

    for frame_number in range(reference_frame_number + 1, total_frames):
        ok, current_frame = cap.read()

        if not ok:
            break

        current_gray = prepare_tracking_gray(current_frame)

        tracking_mask = build_tracking_mask(
            anchor_gray.shape,
            regions,
            reference_to_anchor,
            margin_pixels=args.tracking_margin,
        )

        result, reason = estimate_current_transform(
            anchor_gray,
            current_gray,
            tracking_mask,
            reference_to_anchor,
            width,
            height,
            args,
        )

        if result is not None:
            current_h = result["reference_to_current"]
            transforms[frame_number] = current_h

            logs[frame_number] = {
                "status": "TRACKED",
                "method": result["method"],
                "matched_points": result["matched_points"],
                "inlier_count": result["inlier_count"],
                "inlier_ratio": result["inlier_ratio"],
                "reason": reason,
            }

            # Only a trusted frame becomes the next anchor.
            anchor_gray = current_gray
            reference_to_anchor = current_h
            failed_since_anchor = 0

        else:
            # Hold old polygons temporarily and keep trying to recover
            # from the same last-good anchor on upcoming frames.
            transforms[frame_number] = reference_to_anchor.copy()
            failed_since_anchor += 1

            status = "HOLD_LAST_GOOD"

            if failed_since_anchor >= args.scene_change_warning_frames:
                status = "SCENE_CHANGE_SUSPECTED"

            logs[frame_number] = {
                "status": status,
                "method": "NONE",
                "matched_points": 0,
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "reason": reason,
            }

        if frame_number % 100 == 0 or frame_number == total_frames - 1:
            pct = 100.0 * (frame_number + 1) / total_frames
            print(
                f"\rTracking: {frame_number + 1}/{total_frames} ({pct:.1f}%)",
                end="",
                flush=True,
            )

    print()
    cap.release()

    return {
        "transforms": transforms,
        "logs": logs,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "reference_frame_number": reference_frame_number,
    }


# ============================================================
# SAVE CSV + JSON + PKL
# ============================================================

def save_tracking_data(output_dir, video_path, regions_data, result):
    """
    Save both the motion transforms AND actual per-frame polygon
    coordinates. This is what the thermal processor can consume later.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms = result["transforms"]
    logs = result["logs"]
    fps = result["fps"]
    width = result["width"]
    height = result["height"]
    total_frames = result["total_frames"]

    csv_path = output_dir / "motion_log.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "frame",
            "timestamp_seconds",
            "status",
            "method",
            "matched_points",
            "inlier_count",
            "inlier_ratio",
            "reason",
            "camera_dx_pixels",
            "camera_dy_pixels",
            "camera_translation_pixels",
            "camera_rotation_deg",
            "camera_scale",
            "H00", "H01", "H02",
            "H10", "H11", "H12",
        ])

        for frame_number in range(total_frames):
            h = transforms.get(frame_number, identity_h())
            metrics = affine_metrics(h, width, height)
            log = logs.get(frame_number, {})

            writer.writerow([
                frame_number,
                frame_number / fps if fps > 0 else "",
                log.get("status", "MISSING"),
                log.get("method", "NONE"),
                log.get("matched_points", 0),
                log.get("inlier_count", 0),
                log.get("inlier_ratio", 0.0),
                log.get("reason", "MISSING"),
                metrics["dx"],
                metrics["dy"],
                metrics["translation"],
                metrics["rotation_deg"],
                metrics["scale"],
                h[0, 0], h[0, 1], h[0, 2],
                h[1, 0], h[1, 1], h[1, 2],
            ])

    # JSON/PKL contain transformed ROI points for every frame.
    frame_records = []

    for frame_number in range(total_frames):
        h = transforms.get(frame_number, identity_h())
        metrics = affine_metrics(h, width, height)

        frame_regions = []

        for region in regions_data["regions"]:
            moved = transform_points(region["pixel_points"], h)
            pixel_points = np.rint(moved).astype(int).tolist()

            normalized_points = [
                [point[0] / width, point[1] / height]
                for point in pixel_points
            ]

            frame_regions.append({
                "id": region["id"],
                "name": region["name"],
                "shape_type": region["shape_type"],
                "pixel_points": pixel_points,
                "normalized_points": normalized_points,
            })

        frame_records.append({
            "frame": frame_number,
            "timestamp_seconds": frame_number / fps if fps > 0 else None,
            "status": logs.get(frame_number, {}).get("status", "MISSING"),
            "transform_3x3": h.tolist(),
            "camera_motion": metrics,
            "regions": frame_regions,
        })

    tracking_data = {
        "video": {
            "filename": Path(video_path).name,
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "reference_frame_number": result["reference_frame_number"],
        },
        "reference_regions": regions_data["regions"],
        "frames": frame_records,
    }

    json_path = output_dir / "tracked_regions.json"
    pkl_path = output_dir / "tracked_regions.pkl"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tracking_data, f, indent=2)

    with open(pkl_path, "wb") as f:
        pickle.dump(tracking_data, f)

    return csv_path, json_path, pkl_path


# ============================================================
# PASS 2: RENDER MOVING POLYGONS
# ============================================================

def write_debug_video(video_path, output_dir, regions_data, result, show_fixed):
    """
    RED    = automatically compensated ROI
    YELLOW = original fixed ROI (only with --show-fixed-polygons)

    The yellow/red comparison is especially useful on a synthetic bump.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = result["fps"]
    width = result["width"]
    height = result["height"]
    total_frames = result["total_frames"]

    output_path = Path(output_dir) / "motion_compensated_video.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    print("\nPASS 2: Writing motion-compensated ROI video...")

    for frame_number in range(total_frames):
        ok, frame = cap.read()

        if not ok:
            break

        h = result["transforms"].get(frame_number, identity_h())

        # Optional original static ROI, for visual proof/debugging.
        if show_fixed:
            for region in regions_data["regions"]:
                fixed = np.asarray(region["pixel_points"], dtype=np.int32)
                cv2.polylines(frame, [fixed], True, YELLOW, 1)

        # Motion-compensated polygons.
        for region in regions_data["regions"]:
            moved = transform_points(region["pixel_points"], h)
            polygon = np.rint(moved).astype(np.int32)

            cv2.polylines(frame, [polygon], True, RED, LINE_THICKNESS)

            min_x = int(np.min(polygon[:, 0]))
            min_y = int(np.min(polygon[:, 1]))

            cv2.putText(
                frame,
                region["name"],
                (max(0, min_x), max(25, min_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                RED,
                2,
                cv2.LINE_AA,
            )

        metrics = affine_metrics(h, width, height)
        log = result["logs"].get(frame_number, {})

        lines = [
            (
                f"Camera motion: dx={metrics['dx']:+.1f}px  "
                f"dy={metrics['dy']:+.1f}px  "
                f"rot={metrics['rotation_deg']:+.2f}deg  "
                f"scale={metrics['scale']:.4f}"
            ),
            (
                f"State: {log.get('status', 'UNKNOWN')} | "
                f"Method: {log.get('method', 'NONE')}"
            ),
        ]

        for index, line in enumerate(lines):
            y = 32 + index * 28
            (text_w, text_h), baseline = cv2.getTextSize(
                line,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
            )

            cv2.rectangle(
                frame,
                (10, y - text_h - 6),
                (18 + text_w, y + baseline + 4),
                BLACK,
                -1,
            )

            cv2.putText(
                frame,
                line,
                (14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                WHITE,
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)

        if frame_number % 100 == 0 or frame_number == total_frames - 1:
            pct = 100.0 * (frame_number + 1) / total_frames
            print(
                f"\rWriting: {frame_number + 1}/{total_frames} ({pct:.1f}%)",
                end="",
                flush=True,
            )

    print()
    cap.release()
    writer.release()

    return output_path


# ============================================================
# SUMMARY
# ============================================================

def save_summary(output_dir, result):
    """Save compact tracking-health statistics."""
    status_counts = Counter(log["status"] for log in result["logs"].values())
    method_counts = Counter(log["method"] for log in result["logs"].values())

    metrics = [
        affine_metrics(h, result["width"], result["height"])
        for h in result["transforms"].values()
    ]

    summary = {
        "total_frames": result["total_frames"],
        "reference_frame_number": result["reference_frame_number"],
        "status_counts": dict(status_counts),
        "method_counts": dict(method_counts),
        "maximum_absolute_dx_pixels": max((abs(m["dx"]) for m in metrics), default=0.0),
        "maximum_absolute_dy_pixels": max((abs(m["dy"]) for m in metrics), default=0.0),
        "maximum_translation_pixels": max((m["translation"] for m in metrics), default=0.0),
        "maximum_absolute_rotation_deg": max((abs(m["rotation_deg"]) for m in metrics), default=0.0),
        "minimum_scale": min((m["scale"] for m in metrics), default=1.0),
        "maximum_scale": max((m["scale"] for m in metrics), default=1.0),
    }

    path = Path(output_dir) / "motion_summary.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return path


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase A: detect small accidental thermal-camera movement "
            "and automatically move saved ROI polygons."
        )
    )

    parser.add_argument("video", help="Path to thermal MP4")
    parser.add_argument("regions_json", help="Path to Stage-1 regions.json")

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root. A video-name folder is created automatically.",
    )

    # Feature-selection settings.
    parser.add_argument("--tracking-margin", type=int, default=80)
    parser.add_argument("--max-corners", type=int, default=900)
    parser.add_argument("--min-corner-distance", type=float, default=7.0)

    # RANSAC quality requirements.
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.45)

    # Maximum movement allowed between last-good anchor and current frame.
    parser.add_argument("--max-delta-translation", type=float, default=100.0)
    parser.add_argument("--max-delta-rotation", type=float, default=5.0)
    parser.add_argument("--min-delta-scale", type=float, default=0.92)
    parser.add_argument("--max-delta-scale", type=float, default=1.08)

    # Maximum overall movement that we still classify as "accidental".
    parser.add_argument("--max-cumulative-translation", type=float, default=220.0)
    parser.add_argument("--max-cumulative-rotation", type=float, default=10.0)
    parser.add_argument("--min-cumulative-scale", type=float, default=0.85)
    parser.add_argument("--max-cumulative-scale", type=float, default=1.15)

    parser.add_argument(
        "--scene-change-warning-frames",
        type=int,
        default=20,
        help=(
            "After this many consecutive unrecoverable frames, mark "
            "SCENE_CHANGE_SUSPECTED. Phase B will later handle that case."
        ),
    )

    parser.add_argument(
        "--show-fixed-polygons",
        action="store_true",
        help=(
            "Also draw the old static polygons in yellow. Useful for "
            "visually proving that the red compensated polygons move."
        ),
    )

    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    regions_path = Path(args.regions_json).resolve()
    regions_data = load_json(regions_path)

    # Validate video resolution against the Stage-1 annotations.
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    ann_width = int(regions_data["video"]["width"])
    ann_height = int(regions_data["video"]["height"])

    if (width, height) != (ann_width, ann_height):
        raise RuntimeError(
            "Video resolution does not match regions.json.\n"
            f"Video       : {width} x {height}\n"
            f"Annotations : {ann_width} x {ann_height}"
        )

    output_dir = build_video_output_dir(video_path, args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n============================================")
    print("PHASE A - ACCIDENTAL CAMERA MOVEMENT")
    print("============================================")
    print(f"Video       : {video_path}")
    print(f"Annotations : {regions_path}")
    print(f"Resolution  : {width} x {height}")
    print(f"FPS         : {fps:.3f}")
    print(f"Frames      : {total_frames}")
    print(f"Output      : {output_dir}")

    result = compute_transforms(video_path, regions_data, args)

    csv_path, json_path, pkl_path = save_tracking_data(
        output_dir,
        video_path,
        regions_data,
        result,
    )

    summary_path = save_summary(output_dir, result)

    video_output = write_debug_video(
        video_path,
        output_dir,
        regions_data,
        result,
        show_fixed=args.show_fixed_polygons,
    )

    print("\n============================================")
    print("PHASE A COMPLETE")
    print("============================================")
    print(f"Motion video : {video_output}")
    print(f"Motion CSV   : {csv_path}")
    print(f"Tracked JSON : {json_path}")
    print(f"Tracked PKL  : {pkl_path}")
    print(f"Summary      : {summary_path}")


if __name__ == "__main__":
    main()
