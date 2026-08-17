import argparse
from pathlib import Path

import cv2


# ============================================================
# SYNTHETIC ACCIDENTAL CAMERA MOVEMENT GENERATOR
# ============================================================
#
# This is only a geometric test utility.
#
# If you do not yet have a real clip where the thermal camera gets
# nudged, this script creates one from a known-good static video.
# It applies a small translation / rotation / scale change after a
# chosen timestamp while keeping the same video duration and FPS.
#
# NOTE:
# A real thermal camera often redraws its UI overlays after the sensor
# view moves. This simulator transforms the entire rendered frame,
# including UI. That is not a perfect camera-firmware simulation, but
# it is excellent for checking whether Phase A can recover a known
# 20-pixel / 1-degree geometric shift and move the polygons with it.
# ============================================================


def build_affine(width, height, dx, dy, rotation_deg, scale):
    """Build an affine transform around image center plus x/y shift."""
    center = (width / 2.0, height / 2.0)

    matrix = cv2.getRotationMatrix2D(
        center,
        float(rotation_deg),
        float(scale),
    )

    matrix[0, 2] += float(dx)
    matrix[1, 2] += float(dy)

    return matrix


def motion_at_frame(frame_number, move_frame, transition_frames, dx, dy, rotation, scale):
    """
    Ramp motion over a few frames instead of teleporting instantly.

    transition_frames=1 -> abrupt bump
    transition_frames=4 -> short physical-looking movement
    """
    if frame_number < move_frame:
        fraction = 0.0
    else:
        fraction = min(
            1.0,
            (frame_number - move_frame + 1) / max(1, transition_frames),
        )

    return (
        dx * fraction,
        dy * fraction,
        rotation * fraction,
        1.0 + (scale - 1.0) * fraction,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a synthetic MP4 containing a small accidental "
            "camera movement for Phase-A testing."
        )
    )

    parser.add_argument("video", help="Source MP4")

    parser.add_argument(
        "--output",
        default=None,
        help="Output MP4. Default: <source>_synthetic_camera_bump.mp4",
    )

    parser.add_argument("--move-at-seconds", type=float, default=4.0)
    parser.add_argument("--dx", type=float, default=20.0)
    parser.add_argument("--dy", type=float, default=-10.0)
    parser.add_argument("--rotation", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--transition-frames", type=int, default=4)

    args = parser.parse_args()

    video_path = Path(args.video).resolve()

    if args.output is None:
        output_path = video_path.with_name(
            video_path.stem + "_synthetic_camera_bump.mp4"
        )
    else:
        output_path = Path(args.output).resolve()

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise RuntimeError("Video reported invalid FPS.")

    move_frame = int(round(args.move_at_seconds * fps))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    print("\nCreating synthetic accidental-camera test...")
    print(f"Source     : {video_path}")
    print(f"Output     : {output_path}")
    print(
        f"Final move : dx={args.dx:+.1f}px, dy={args.dy:+.1f}px, "
        f"rotation={args.rotation:+.2f}deg, scale={args.scale:.4f}"
    )
    print(
        f"Starts at  : {args.move_at_seconds:.2f}s "
        f"(frame {move_frame})"
    )

    for frame_number in range(total_frames):
        ok, frame = cap.read()

        if not ok:
            break

        dx, dy, rotation, scale = motion_at_frame(
            frame_number,
            move_frame,
            args.transition_frames,
            args.dx,
            args.dy,
            args.rotation,
            args.scale,
        )

        if (
            abs(dx) > 1e-9
            or abs(dy) > 1e-9
            or abs(rotation) > 1e-9
            or abs(scale - 1.0) > 1e-9
        ):
            matrix = build_affine(
                width,
                height,
                dx,
                dy,
                rotation,
                scale,
            )

            frame = cv2.warpAffine(
                frame,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
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

    print(f"\nSynthetic test video saved:\n{output_path}")


if __name__ == "__main__":
    main()
