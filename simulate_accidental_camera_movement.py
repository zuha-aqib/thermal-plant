import argparse
from pathlib import Path

import cv2

import step_code as ps


DEFAULT_OUTPUT_ROOT = "synthetic-moved-videos"
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


def build_variant_dir(video_path, output_root):
    """Create one folder per source video so it can hold many variants.

    Example:
        raw-videos/Furnace/Camera-01/sample.mp4

    becomes:
        synthetic-moved-videos/Furnace/Camera-01/sample/
            transformations.json
            accidental_001.mp4
            accidental_002.mp4
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


def build_legacy_output_path(video_path, output_root):
    """Return the path used by the old single-output simulator."""
    video_path = Path(video_path).resolve()
    output_root = Path(output_root).resolve()
    raw_root = find_raw_videos_ancestor(video_path)

    if raw_root is not None:
        return output_root / video_path.relative_to(raw_root)

    return output_root / video_path.name


def load_transformations(path, source_video):
    """Load the variant ledger, creating an empty structure if needed."""
    data = ps.load_json(path, default=None)

    if not isinstance(data, dict):
        data = {
            "manifest_version": 1,
            "generator": "synthetic_accidental_camera_movement",
            "source_video_name": Path(source_video).name,
            "variants": [],
        }

    if not isinstance(data.get("variants"), list):
        data["variants"] = []

    return data


def next_variant_number(variants):
    """Return the next monotonically increasing accidental variant number."""
    highest = 0

    for variant in variants:
        variant_id = str(variant.get("id", ""))

        if variant_id.startswith("accidental_"):
            try:
                highest = max(highest, int(variant_id.split("_")[-1]))
            except ValueError:
                pass

    return highest + 1


def normalize_parameters(args):
    """Store only settings that actually define the synthetic movement."""
    return {
        "move_at_seconds": float(args.move_at_seconds),
        "dx_pixels": float(args.dx),
        "dy_pixels": float(args.dy),
        "rotation_degrees": float(args.rotation),
        "scale": float(args.scale),
        "transition_frames": int(args.transition_frames),
    }


def find_exact_variant(variants, signature):
    """Find a previously registered variant with the same exact signature."""
    for variant in variants:
        if variant.get("signature") == signature:
            return variant

    return None


# ============================================================
# SYNTHETIC MOVEMENT
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
    """Ramp the bump over a few frames instead of teleporting instantly."""
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


def create_variant(video_path, output_path, parameters):
    """Generate one accidental-camera synthetic video."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError("Video reported invalid FPS/frame count.")

    move_frame = int(round(parameters["move_at_seconds"] * fps))

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
    print(f"Working out: {output_path}")
    print(
        "Final move : "
        f"dx={parameters['dx_pixels']:+.1f}px, "
        f"dy={parameters['dy_pixels']:+.1f}px, "
        f"rotation={parameters['rotation_degrees']:+.2f}deg, "
        f"scale={parameters['scale']:.4f}"
    )
    print(
        f"Starts at  : {parameters['move_at_seconds']:.2f}s "
        f"(frame {move_frame})"
    )

    try:
        for frame_number in range(total_frames):
            ok, frame = cap.read()

            if not ok:
                break

            dx, dy, rotation, scale = motion_at_frame(
                frame_number,
                move_frame,
                parameters["transition_frames"],
                parameters["dx_pixels"],
                parameters["dy_pixels"],
                parameters["rotation_degrees"],
                parameters["scale"],
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

    finally:
        cap.release()
        writer.release()

    print()

    if not ps.file_exists_and_nonempty(output_path):
        raise RuntimeError("Synthetic video was not written successfully.")


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create one or more tracked synthetic accidental-camera "
            "movement variants from a source MP4."
        )
    )

    parser.add_argument("video", help="Source MP4")

    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Synthetic output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )

    parser.add_argument("--move-at-seconds", type=float, default=4.0)
    parser.add_argument("--dx", type=float, default=20.0)
    parser.add_argument("--dy", type=float, default=-10.0)
    parser.add_argument("--rotation", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--transition-frames", type=int, default=4)

    args = parser.parse_args()

    video_path = Path(args.video).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    if args.transition_frames <= 0:
        raise ValueError("--transition-frames must be greater than 0.")

    if args.scale <= 0:
        raise ValueError("--scale must be greater than 0.")

    variant_dir = build_variant_dir(video_path, args.output_root)
    variant_dir.mkdir(parents=True, exist_ok=True)

    transformations_path = variant_dir / TRANSFORMATIONS_FILENAME
    ledger = load_transformations(transformations_path, video_path)

    # Migration safety: older versions wrote one MP4 directly under the
    # mirrored parent folder and stored no transformation metadata.  We can
    # preserve it, but we cannot truthfully claim which parameters created it.
    legacy_output = build_legacy_output_path(video_path, args.output_root)

    if (
        not ledger["variants"]
        and ps.file_exists_and_nonempty(legacy_output)
    ):
        print("\nLEGACY / UNTRACKED SYNTHETIC VIDEO FOUND")
        print(f"Old output: {legacy_output}")
        print(
            "The old simulator did not save transformation parameters, so "
            "the script cannot prove whether it matches the command you are "
            "requesting now. The legacy MP4 will not be overwritten."
        )

        if not ps.prompt_yes_no(
            "Create a new tracked variant for the requested transformation?",
            default=True,
        ):
            print("Skipped. Legacy output was not changed.")
            return

    print("\nChecking source-video fingerprint...")
    source_fingerprint = ps.fingerprint_video(video_path)
    parameters = normalize_parameters(args)

    signature_payload = {
        "source_video_sha256": source_fingerprint["sha256"],
        "parameters": parameters,
    }
    signature = ps.manifest_signature(signature_payload)

    exact_variant = find_exact_variant(ledger["variants"], signature)

    if exact_variant is not None:
        variant_id = exact_variant["id"]
        output_path = variant_dir / exact_variant["output_filename"]

        print("\nEXACT SYNTHETIC VARIANT ALREADY REGISTERED")
        print(f"Variant      : {variant_id}")
        print(f"Output       : {output_path}")
        print(f"Move at      : {parameters['move_at_seconds']:.2f} s")
        print(f"X shift      : {parameters['dx_pixels']:+.1f} px")
        print(f"Y shift      : {parameters['dy_pixels']:+.1f} px")
        print(f"Rotation     : {parameters['rotation_degrees']:+.2f} deg")
        print(f"Scale        : {parameters['scale']:.4f}")
        print(f"Transition   : {parameters['transition_frames']} frames")

        output_exists = ps.file_exists_and_nonempty(output_path)

        if output_exists:
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
        old_source_hashes = {
            variant.get("source_video", {}).get("sha256")
            for variant in ledger["variants"]
            if variant.get("source_video", {}).get("sha256")
        }

        if old_source_hashes and source_fingerprint["sha256"] not in old_source_hashes:
            print(
                "\nSource video content has changed since older variants were made. "
                "The new source hash will therefore create a new variant."
            )

        number = next_variant_number(ledger["variants"])
        variant_id = f"accidental_{number:03d}"
        output_path = variant_dir / f"{variant_id}.mp4"

        print("\nNEW SYNTHETIC TRANSFORMATION")
        print(f"Existing variants: {len(ledger['variants'])}")
        print(f"Creating         : {variant_id}")
        print(f"Output           : {output_path}")

    temp_path = output_path.with_name(
        output_path.stem + ".__processing__" + output_path.suffix
    )
    ps.safe_remove(temp_path)

    try:
        create_variant(
            video_path=video_path,
            output_path=temp_path,
            parameters=parameters,
        )

        ps.replace_completed_file(temp_path, output_path)

        record = {
            "id": variant_id,
            "signature": signature,
            "created_at_utc": ps.now_utc_iso(),
            "output_filename": output_path.name,
            "source_video": source_fingerprint,
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

    print("\nSynthetic accidental variant completed.")
    print(f"Variant ledger : {transformations_path}")
    print(f"Saved video    : {output_path}")


if __name__ == "__main__":
    main()
