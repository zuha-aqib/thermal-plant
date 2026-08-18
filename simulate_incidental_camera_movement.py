import argparse
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# SYNTHETIC INCIDENTAL / INTENTIONAL CAMERA MOVEMENT TEST
# ============================================================
#
# Creates a simple Phase-B test video using TWO different source
# videos/views:
#
#   VIEW A stable
#       -> simulated camera pan
#   VIEW B stable
#
# Optionally:
#
#   VIEW B stable
#       -> pan back
#   VIEW A stable again
#
# This is useful for testing:
#   STABLE -> MOVING -> NEW_STABLE_VIEW -> STABLE
# and, with --return-to-first:
#   MOVING -> RETURN_TO_KNOWN_VIEW
# ============================================================


DEFAULT_OUTPUT_ROOT = "synthetic-incidental-videos"



def find_raw_videos_ancestor(video_path):
    video_path = Path(video_path).resolve()

    for parent in video_path.parents:
        if parent.name.lower() == "raw-videos":
            return parent

    return None



def build_output_path(video_a, video_b, output_root):
    """
    Save under the same relative parent as video A when possible.
    """
    video_a = Path(video_a).resolve()
    video_b = Path(video_b).resolve()
    output_root = Path(output_root).resolve()

    raw_root = find_raw_videos_ancestor(video_a)

    if raw_root is not None:
        relative_a = video_a.relative_to(raw_root)
        parent = relative_a.parent
    else:
        parent = Path()

    filename = (
        f"{video_a.stem}__TO__{video_b.stem}__phase_b_test.mp4"
    )

    output_path = output_root / parent / filename

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_path



def open_video(path):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {path}"
        )

    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

    if info["fps"] <= 0 or info["frames"] <= 0:
        cap.release()
        raise RuntimeError(
            f"Video reports invalid FPS/frame count: {path}"
        )

    return cap, info



def read_frame_at_time(cap, info, seconds, target_size):
    """
    Read the source frame closest to a requested timestamp.
    Loop inside the source video if the requested segment is longer
    than the available footage.
    """
    duration = info["frames"] / info["fps"]

    if duration <= 0:
        source_time = 0.0
    else:
        source_time = seconds % duration

    source_frame = int(
        round(source_time * info["fps"])
    )

    source_frame = max(
        0,
        min(source_frame, info["frames"] - 1),
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        source_frame,
    )

    ok, frame = cap.read()

    if not ok:
        raise RuntimeError(
            f"Could not read source frame {source_frame}"
        )

    target_width, target_height = target_size

    if frame.shape[1] != target_width or frame.shape[0] != target_height:
        frame = cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )

    return frame



def slide_transition(frame_a, frame_b, fraction):
    """
    Simulate a horizontal PTZ-style pan.

    At fraction 0:
        frame A fills the screen.

    At fraction 1:
        frame B fills the screen.

    During transition:
        A slides left while B enters from the right.
    """
    fraction = float(
        np.clip(fraction, 0.0, 1.0)
    )

    height, width = frame_a.shape[:2]

    offset = int(round(width * fraction))

    canvas = np.zeros_like(frame_a)

    # Remaining right-side portion of A.
    a_visible_width = width - offset

    if a_visible_width > 0:
        canvas[:, :a_visible_width] = (
            frame_a[:, offset:]
        )

    # Left-side portion of B enters from the right.
    if offset > 0:
        canvas[:, a_visible_width:] = (
            frame_b[:, :offset]
        )

    return canvas



def write_stable_segment(
    writer,
    cap,
    info,
    seconds,
    output_fps,
    target_size,
    source_time_offset=0.0,
):
    frame_count = int(round(seconds * output_fps))

    for index in range(frame_count):
        source_time = (
            source_time_offset
            + index / output_fps
        )

        frame = read_frame_at_time(
            cap,
            info,
            source_time,
            target_size,
        )

        writer.write(frame)

    return frame_count



def write_transition_segment(
    writer,
    cap_a,
    info_a,
    cap_b,
    info_b,
    seconds,
    output_fps,
    target_size,
    source_time_a,
    source_time_b,
):
    frame_count = max(
        1,
        int(round(seconds * output_fps)),
    )

    for index in range(frame_count):
        if frame_count == 1:
            fraction = 1.0
        else:
            fraction = index / (frame_count - 1)

        frame_a = read_frame_at_time(
            cap_a,
            info_a,
            source_time_a + index / output_fps,
            target_size,
        )

        frame_b = read_frame_at_time(
            cap_b,
            info_b,
            source_time_b + index / output_fps,
            target_size,
        )

        output = slide_transition(
            frame_a,
            frame_b,
            fraction,
        )

        writer.write(output)

    return frame_count



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a synthetic Phase-B video that moves from one "
            "stable camera view to another."
        )
    )

    parser.add_argument(
        "video_a",
        help="First stable source video/view",
    )

    parser.add_argument(
        "video_b",
        help="Second stable source video/view",
    )

    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            f"Output root. Default: {DEFAULT_OUTPUT_ROOT}"
        ),
    )

    parser.add_argument(
        "--stable-seconds",
        type=float,
        default=4.0,
        help=(
            "Length of each stable section. Default: 4 seconds."
        ),
    )

    parser.add_argument(
        "--transition-seconds",
        type=float,
        default=1.2,
        help=(
            "Length of simulated camera pan. Default: 1.2 seconds."
        ),
    )

    parser.add_argument(
        "--return-to-first",
        action="store_true",
        help=(
            "After VIEW B, pan back to VIEW A. Useful for testing "
            "recognition of a previously seen stable view."
        ),
    )

    args = parser.parse_args()

    video_a = Path(args.video_a).resolve()
    video_b = Path(args.video_b).resolve()

    output_path = build_output_path(
        video_a,
        video_b,
        args.output_root,
    )

    cap_a, info_a = open_video(video_a)
    cap_b, info_b = open_video(video_b)

    output_fps = info_a["fps"]
    width = info_a["width"]
    height = info_a["height"]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )

    if not writer.isOpened():
        cap_a.release()
        cap_b.release()
        raise RuntimeError(
            f"Could not create output: {output_path}"
        )

    target_size = (width, height)

    print("\nCreating synthetic incidental-camera test...")
    print(f"VIEW A     : {video_a}")
    print(f"VIEW B     : {video_b}")
    print(f"Output     : {output_path}")
    print(f"Stable     : {args.stable_seconds:.2f}s per view")
    print(f"Transition : {args.transition_seconds:.2f}s")
    print(f"Return A   : {args.return_to_first}")

    total_written = 0

    # A stable.
    total_written += write_stable_segment(
        writer,
        cap_a,
        info_a,
        seconds=args.stable_seconds,
        output_fps=output_fps,
        target_size=target_size,
        source_time_offset=0.0,
    )

    # A -> B movement.
    total_written += write_transition_segment(
        writer,
        cap_a,
        info_a,
        cap_b,
        info_b,
        seconds=args.transition_seconds,
        output_fps=output_fps,
        target_size=target_size,
        source_time_a=args.stable_seconds,
        source_time_b=0.0,
    )

    # B stable.
    total_written += write_stable_segment(
        writer,
        cap_b,
        info_b,
        seconds=args.stable_seconds,
        output_fps=output_fps,
        target_size=target_size,
        source_time_offset=args.transition_seconds,
    )

    if args.return_to_first:
        # B -> A movement.
        total_written += write_transition_segment(
            writer,
            cap_b,
            info_b,
            cap_a,
            info_a,
            seconds=args.transition_seconds,
            output_fps=output_fps,
            target_size=target_size,
            source_time_a=(
                args.stable_seconds
                + args.transition_seconds
            ),
            source_time_b=(
                args.stable_seconds
                + args.transition_seconds
            ),
        )

        # A stable again.
        total_written += write_stable_segment(
            writer,
            cap_a,
            info_a,
            seconds=args.stable_seconds,
            output_fps=output_fps,
            target_size=target_size,
            source_time_offset=(
                args.stable_seconds
                + 2 * args.transition_seconds
            ),
        )

    writer.release()
    cap_a.release()
    cap_b.release()

    duration = total_written / output_fps

    print(f"\nFrames written : {total_written}")
    print(f"Duration       : {duration:.2f}s")
    print(f"Saved          : {output_path}")


if __name__ == "__main__":
    main()
