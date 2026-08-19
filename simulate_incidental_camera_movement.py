import argparse
from pathlib import Path

import cv2
import numpy as np

import step_code as ps


DEFAULT_OUTPUT_ROOT = "synthetic-incidental-videos"
TRANSFORMATIONS_FILENAME = "transformations.json"


# ============================================================
# PATH / VARIANT HELPERS
# ============================================================


def find_raw_videos_ancestor(video_path):
    """Find the nearest parent directory named raw-videos."""
    video_path = Path(video_path).resolve()

    for parent in video_path.parents:
        if parent.name.lower() == "raw-videos":
            return parent

    return None


def build_variant_dir(video_a, video_b, output_root):
    """Keep all A->B Phase-B variants together in one folder."""
    video_a = Path(video_a).resolve()
    video_b = Path(video_b).resolve()
    output_root = Path(output_root).resolve()

    raw_root = find_raw_videos_ancestor(video_a)

    if raw_root is not None:
        relative_a = video_a.relative_to(raw_root)
        relative_parent = relative_a.parent
    else:
        relative_parent = Path()

    pair_name = f"{video_a.stem}__TO__{video_b.stem}"
    return output_root / relative_parent / pair_name


def build_legacy_output_path(video_a, video_b, output_root):
    """Return the old single-output Phase-B simulator path."""
    video_a = Path(video_a).resolve()
    video_b = Path(video_b).resolve()
    output_root = Path(output_root).resolve()
    raw_root = find_raw_videos_ancestor(video_a)

    if raw_root is not None:
        relative_a = video_a.relative_to(raw_root)
        relative_parent = relative_a.parent
    else:
        relative_parent = Path()

    filename = f"{video_a.stem}__TO__{video_b.stem}__phase_b_test.mp4"
    return output_root / relative_parent / filename


def load_transformations(path, video_a, video_b):
    """Load/create the Phase-B synthetic variant ledger."""
    data = ps.load_json(path, default=None)

    if not isinstance(data, dict):
        data = {
            "manifest_version": 1,
            "generator": "synthetic_incidental_camera_movement",
            "source_a_name": Path(video_a).name,
            "source_b_name": Path(video_b).name,
            "variants": [],
        }

    if not isinstance(data.get("variants"), list):
        data["variants"] = []

    return data


def next_variant_number(variants):
    """Return the next monotonically increasing incidental variant number."""
    highest = 0

    for variant in variants:
        variant_id = str(variant.get("id", ""))

        if variant_id.startswith("incidental_"):
            try:
                highest = max(highest, int(variant_id.split("_")[-1]))
            except ValueError:
                pass

    return highest + 1


def normalize_parameters(args):
    """Return only parameters that define the generated Phase-B sequence."""
    return {
        "stable_seconds": float(args.stable_seconds),
        "transition_seconds": float(args.transition_seconds),
        "return_to_first": bool(args.return_to_first),
    }


def find_exact_variant(variants, signature):
    """Find a registered variant with the exact source+parameter signature."""
    for variant in variants:
        if variant.get("signature") == signature:
            return variant

    return None


# ============================================================
# VIDEO HELPERS
# ============================================================


def open_video(path):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

    if info["fps"] <= 0 or info["frames"] <= 0:
        cap.release()
        raise RuntimeError(f"Video reports invalid FPS/frame count: {path}")

    return cap, info


def read_frame_at_time(cap, info, seconds, target_size):
    """Read nearest source frame, looping when the requested segment is longer."""
    duration = info["frames"] / info["fps"]
    source_time = 0.0 if duration <= 0 else seconds % duration

    source_frame = int(round(source_time * info["fps"]))
    source_frame = max(0, min(source_frame, info["frames"] - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
    ok, frame = cap.read()

    if not ok:
        raise RuntimeError(f"Could not read source frame {source_frame}")

    target_width, target_height = target_size

    if frame.shape[1] != target_width or frame.shape[0] != target_height:
        frame = cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )

    return frame


def slide_transition(frame_a, frame_b, fraction):
    """Simulate a horizontal PTZ-style pan from A to B."""
    fraction = float(np.clip(fraction, 0.0, 1.0))

    height, width = frame_a.shape[:2]
    offset = int(round(width * fraction))
    canvas = np.zeros_like(frame_a)

    a_visible_width = width - offset

    if a_visible_width > 0:
        canvas[:, :a_visible_width] = frame_a[:, offset:]

    if offset > 0:
        canvas[:, a_visible_width:] = frame_b[:, :offset]

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
        source_time = source_time_offset + index / output_fps

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
    frame_count = max(1, int(round(seconds * output_fps)))

    for index in range(frame_count):
        fraction = 1.0 if frame_count == 1 else index / (frame_count - 1)

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

        writer.write(slide_transition(frame_a, frame_b, fraction))

    return frame_count


def create_variant(video_a, video_b, output_path, parameters):
    """Generate one tracked A -> B (optionally -> A) synthetic sequence."""
    cap_a, info_a = open_video(video_a)
    cap_b, info_b = open_video(video_b)

    output_fps = info_a["fps"]
    width = info_a["width"]
    height = info_a["height"]
    target_size = (width, height)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )

    if not writer.isOpened():
        cap_a.release()
        cap_b.release()
        raise RuntimeError(f"Could not create output: {output_path}")

    print("\nCreating synthetic incidental-camera test...")
    print(f"VIEW A      : {video_a}")
    print(f"VIEW B      : {video_b}")
    print(f"Working out : {output_path}")
    print(f"Stable      : {parameters['stable_seconds']:.2f}s per view")
    print(f"Transition  : {parameters['transition_seconds']:.2f}s")
    print(f"Return A    : {parameters['return_to_first']}")

    total_written = 0

    try:
        # A stable.
        total_written += write_stable_segment(
            writer,
            cap_a,
            info_a,
            seconds=parameters["stable_seconds"],
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
            seconds=parameters["transition_seconds"],
            output_fps=output_fps,
            target_size=target_size,
            source_time_a=parameters["stable_seconds"],
            source_time_b=0.0,
        )

        # B stable.
        total_written += write_stable_segment(
            writer,
            cap_b,
            info_b,
            seconds=parameters["stable_seconds"],
            output_fps=output_fps,
            target_size=target_size,
            source_time_offset=parameters["transition_seconds"],
        )

        if parameters["return_to_first"]:
            # B -> A movement.
            total_written += write_transition_segment(
                writer,
                cap_b,
                info_b,
                cap_a,
                info_a,
                seconds=parameters["transition_seconds"],
                output_fps=output_fps,
                target_size=target_size,
                source_time_a=(
                    parameters["stable_seconds"]
                    + parameters["transition_seconds"]
                ),
                source_time_b=(
                    parameters["stable_seconds"]
                    + parameters["transition_seconds"]
                ),
            )

            # A stable again.
            total_written += write_stable_segment(
                writer,
                cap_a,
                info_a,
                seconds=parameters["stable_seconds"],
                output_fps=output_fps,
                target_size=target_size,
                source_time_offset=(
                    parameters["stable_seconds"]
                    + 2 * parameters["transition_seconds"]
                ),
            )

    finally:
        writer.release()
        cap_a.release()
        cap_b.release()

    if not ps.file_exists_and_nonempty(output_path):
        raise RuntimeError("Synthetic video was not written successfully.")

    duration = total_written / output_fps
    print(f"\nFrames written : {total_written}")
    print(f"Duration       : {duration:.2f}s")


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create tracked synthetic Phase-B variants that move from "
            "one stable camera view to another."
        )
    )

    parser.add_argument("video_a", help="First stable source video/view")
    parser.add_argument("video_b", help="Second stable source video/view")

    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )

    parser.add_argument(
        "--stable-seconds",
        type=float,
        default=4.0,
        help="Length of each stable section. Default: 4 seconds.",
    )

    parser.add_argument(
        "--transition-seconds",
        type=float,
        default=1.2,
        help="Length of simulated camera pan. Default: 1.2 seconds.",
    )

    parser.add_argument(
        "--return-to-first",
        action="store_true",
        help="After VIEW B, pan back to VIEW A.",
    )

    args = parser.parse_args()

    video_a = Path(args.video_a).resolve()
    video_b = Path(args.video_b).resolve()

    if not video_a.is_file():
        raise FileNotFoundError(video_a)

    if not video_b.is_file():
        raise FileNotFoundError(video_b)

    if args.stable_seconds <= 0:
        raise ValueError("--stable-seconds must be greater than 0.")

    if args.transition_seconds <= 0:
        raise ValueError("--transition-seconds must be greater than 0.")

    variant_dir = build_variant_dir(video_a, video_b, args.output_root)
    variant_dir.mkdir(parents=True, exist_ok=True)

    transformations_path = variant_dir / TRANSFORMATIONS_FILENAME
    ledger = load_transformations(transformations_path, video_a, video_b)

    legacy_output = build_legacy_output_path(
        video_a,
        video_b,
        args.output_root,
    )

    if (
        not ledger["variants"]
        and ps.file_exists_and_nonempty(legacy_output)
    ):
        print("\nLEGACY / UNTRACKED PHASE-B VIDEO FOUND")
        print(f"Old output: {legacy_output}")
        print(
            "The old simulator did not save its stable/transition settings, "
            "so an exact parameter match cannot be proven. The old MP4 will "
            "not be overwritten."
        )

        if not ps.prompt_yes_no(
            "Create a new tracked Phase-B variant for the requested settings?",
            default=True,
        ):
            print("Skipped. Legacy output was not changed.")
            return

    print("\nChecking source-video fingerprints...")
    source_a = ps.fingerprint_video(video_a)
    source_b = ps.fingerprint_video(video_b)
    parameters = normalize_parameters(args)

    signature_payload = {
        "source_a_sha256": source_a["sha256"],
        "source_b_sha256": source_b["sha256"],
        "parameters": parameters,
    }
    signature = ps.manifest_signature(signature_payload)

    exact_variant = find_exact_variant(ledger["variants"], signature)

    if exact_variant is not None:
        variant_id = exact_variant["id"]
        output_path = variant_dir / exact_variant["output_filename"]

        print("\nEXACT PHASE-B SYNTHETIC VARIANT ALREADY REGISTERED")
        print(f"Variant      : {variant_id}")
        print(f"Output       : {output_path}")
        print(f"Stable       : {parameters['stable_seconds']:.2f} s")
        print(f"Transition   : {parameters['transition_seconds']:.2f} s")
        print(f"Return A     : {parameters['return_to_first']}")

        if ps.file_exists_and_nonempty(output_path):
            should_generate = ps.prompt_yes_no(
                "Regenerate this exact variant?",
                default=False,
            )
        else:
            print("The ledger entry exists, but its MP4 is missing/empty.")
            should_generate = ps.prompt_yes_no(
                "Regenerate the missing variant?",
                default=True,
            )

        if not should_generate:
            print("Skipped. Existing synthetic variant was not changed.")
            return

    else:
        prior_source_pairs = {
            (
                variant.get("source_a", {}).get("sha256"),
                variant.get("source_b", {}).get("sha256"),
            )
            for variant in ledger["variants"]
        }

        current_pair = (source_a["sha256"], source_b["sha256"])

        if prior_source_pairs and current_pair not in prior_source_pairs:
            print(
                "\nOne or both source videos changed since older variants "
                "were made. The changed source hashes will create a new variant."
            )

        number = next_variant_number(ledger["variants"])
        variant_id = f"incidental_{number:03d}"
        output_path = variant_dir / f"{variant_id}.mp4"

        print("\nNEW PHASE-B SYNTHETIC CONFIGURATION")
        print(f"Existing variants: {len(ledger['variants'])}")
        print(f"Creating         : {variant_id}")
        print(f"Output           : {output_path}")

    temp_path = output_path.with_name(
        output_path.stem + ".__processing__" + output_path.suffix
    )
    ps.safe_remove(temp_path)

    try:
        create_variant(
            video_a=video_a,
            video_b=video_b,
            output_path=temp_path,
            parameters=parameters,
        )

        ps.replace_completed_file(temp_path, output_path)

        record = {
            "id": variant_id,
            "signature": signature,
            "created_at_utc": ps.now_utc_iso(),
            "output_filename": output_path.name,
            "source_a": source_a,
            "source_b": source_b,
            "parameters": parameters,
            "output": ps.fingerprint_file(output_path, content_hash=True),
        }

        if exact_variant is None:
            ledger["variants"].append(record)
        else:
            index = ledger["variants"].index(exact_variant)
            ledger["variants"][index] = record

        ledger["last_updated_utc"] = ps.now_utc_iso()
        ps.save_json_atomic(transformations_path, ledger)

    except Exception:
        ps.safe_remove(temp_path)
        raise

    print("\nSynthetic incidental variant completed.")
    print(f"Variant ledger : {transformations_path}")
    print(f"Saved video    : {output_path}")


if __name__ == "__main__":
    main()
