import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

DEFAULT_OUTPUT_ROOT = "phase-c-dataset-v0"
CLASS_NAME = "equipment"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def safe_name(text):
    cleaned = "".join(c if c.isalnum() else "_" for c in str(text))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def find_raw_video(raw_root, filename):
    matches = [
        p for p in Path(raw_root).rglob("*")
        if p.is_file() and p.name.lower() == filename.lower()
    ]
    if not matches:
        return None
    if len(matches) > 1:
        print(f"WARNING: multiple files named {filename}; using {matches[0]}")
    return matches[0]


def get_settings(video_stem, config):
    settings = dict(config.get("default", {}))
    settings.update(config.get("videos", {}).get(video_stem, {}))
    return settings


def polygon_area(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_bbox(points):
    pts = np.asarray(points, dtype=np.float64)
    min_x, min_y = np.min(pts[:, 0]), np.min(pts[:, 1])
    max_x, max_y = np.max(pts[:, 0]), np.max(pts[:, 1])
    return [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)]


def flatten_polygon(points):
    return [float(v) for point in points for v in point]


def build_affine(width, height, dx, dy, rotation_deg, scale):
    matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0),
        float(rotation_deg),
        float(scale),
    ).astype(np.float64)
    matrix[0, 2] += float(dx)
    matrix[1, 2] += float(dy)
    return matrix


def transform_polygon(points, matrix):
    pts = np.asarray(points, dtype=np.float64)
    homogeneous = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    return (matrix @ homogeneous.T).T


def polygons_fit_frame(polygons, width, height, border=3):
    for pts in polygons:
        pts = np.asarray(pts, dtype=np.float64)
        if polygon_area(pts) < 10:
            return False
        if np.any(pts[:, 0] < border) or np.any(pts[:, 1] < border):
            return False
        if np.any(pts[:, 0] >= width - border) or np.any(pts[:, 1] >= height - border):
            return False
    return True


def sample_valid_affine(width, height, regions, settings, rng):
    originals = [region["pixel_points"] for region in regions]
    for _ in range(int(settings.get("max_attempts_per_augmentation", 30))):
        dx = rng.uniform(-float(settings["max_dx"]), float(settings["max_dx"]))
        dy = rng.uniform(-float(settings["max_dy"]), float(settings["max_dy"]))
        rotation = rng.uniform(
            -float(settings["max_rotation_deg"]),
            float(settings["max_rotation_deg"]),
        )
        scale = rng.uniform(float(settings["min_scale"]), float(settings["max_scale"]))
        matrix = build_affine(width, height, dx, dy, rotation, scale)
        polygons = [transform_polygon(points, matrix) for points in originals]
        if polygons_fit_frame(polygons, width, height):
            return {
                "matrix": matrix,
                "polygons": polygons,
                "dx": dx,
                "dy": dy,
                "rotation_deg": rotation,
                "scale": scale,
            }
    return None


def add_coco_annotations(coco, image_id, regions, polygons, annotation_id):
    for region, points in zip(regions, polygons):
        points = np.asarray(points, dtype=np.float64)
        coco["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": [flatten_polygon(points)],
            "area": float(polygon_area(points)),
            "bbox": polygon_bbox(points),
            "iscrowd": 0,
            "source_region_id": region.get("id"),
            "source_region_name": region.get("name"),
        })
        annotation_id += 1
    return annotation_id


def save_preview(image, regions, polygons, path):
    preview = image.copy()
    for region, points in zip(regions, polygons):
        poly = np.rint(points).astype(np.int32)
        cv2.polylines(preview, [poly], True, (0, 0, 255), 2)
        x = int(np.min(poly[:, 0]))
        y = int(np.min(poly[:, 1]))
        cv2.putText(
            preview,
            str(region.get("name", CLASS_NAME)),
            (max(0, x), max(22, y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), preview)


def process_video(
    video_path,
    regions_json,
    images_dir,
    previews_dir,
    coco,
    manifest_writer,
    image_id,
    annotation_id,
    sample_every_seconds,
    settings,
    rng,
    preview_every,
):
    data = load_json(regions_json)
    regions = data.get("regions", [])
    if not regions:
        print(f"Skipping {video_path.name}: no regions")
        return image_id, annotation_id, 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS for {video_path}")

    step = max(1, int(round(fps * float(sample_every_seconds))))
    aug_count = int(settings.get("augmentations_per_frame", 0))
    safe_stem = safe_name(video_path.stem)
    created = 0
    source_index = 0

    print(f"\n{video_path.name}")
    print(f"  sample every {sample_every_seconds:.2f}s")
    print(f"  augmentations/sample: {aug_count}")

    for frame_number in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        if not ok:
            continue

        source_group = f"{safe_stem}__frame_{frame_number:06d}"
        original_polygons = [
            np.asarray(region["pixel_points"], dtype=np.float64)
            for region in regions
        ]

        variants = [
            {
                "kind": "original",
                "image": frame,
                "polygons": original_polygons,
                "dx": 0.0,
                "dy": 0.0,
                "rotation_deg": 0.0,
                "scale": 1.0,
            }
        ]

        for aug_number in range(1, aug_count + 1):
            aug = sample_valid_affine(width, height, regions, settings, rng)
            if aug is None:
                print(f"  warning: could not augment frame {frame_number}")
                continue
            transformed = cv2.warpAffine(
                frame,
                aug["matrix"],
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            aug["kind"] = f"aug_{aug_number:02d}"
            aug["image"] = transformed
            variants.append(aug)

        for variant in variants:
            kind = variant["kind"]
            filename = f"{source_group}__{kind}.jpg"
            cv2.imwrite(
                str(images_dir / filename),
                variant["image"],
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )

            coco["images"].append({
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
                "source_video": video_path.name,
                "source_frame": frame_number,
                "source_group": source_group,
                "augmentation": kind,
            })
            annotation_id = add_coco_annotations(
                coco,
                image_id,
                regions,
                variant["polygons"],
                annotation_id,
            )

            manifest_writer.writerow([
                image_id,
                filename,
                source_group,
                video_path.name,
                frame_number,
                frame_number / fps,
                kind,
                variant.get("dx", 0.0),
                variant.get("dy", 0.0),
                variant.get("rotation_deg", 0.0),
                variant.get("scale", 1.0),
                len(regions),
            ])

            if source_index % max(1, preview_every) == 0:
                save_preview(
                    variant["image"],
                    regions,
                    variant["polygons"],
                    previews_dir / Path(filename).with_suffix(".png"),
                )

            image_id += 1
            created += 1

        source_index += 1

    cap.release()
    return image_id, annotation_id, created


def main():
    parser = argparse.ArgumentParser(
        description="Build a COCO V0 equipment-segmentation dataset from raw thermal videos and Step-1 polygons."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--raw-root", default="raw-videos")
    parser.add_argument("--annotation-root", default="step01-annotate-videos")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", default="phase_c_dataset_config.json")
    parser.add_argument("--sample-every-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-every", type=int, default=10)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    raw_root = project_root / args.raw_root
    annotation_root = project_root / args.annotation_root
    output_root = project_root / args.output
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    config = load_json(config_path)
    images_dir = output_root / "images"
    annotations_dir = output_root / "annotations"
    previews_dir = output_root / "previews"
    metadata_dir = output_root / "metadata"
    for directory in (images_dir, annotations_dir, previews_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    regions_files = sorted(annotation_root.rglob("regions.json"), key=lambda p: str(p).lower())

    coco = {
        "info": {"description": "Thermal equipment instance segmentation dataset V0", "version": "0.1"},
        "licenses": [],
        "categories": [{"id": 1, "name": CLASS_NAME, "supercategory": "industrial_equipment"}],
        "images": [],
        "annotations": [],
    }

    manifest_path = metadata_dir / "dataset_manifest.csv"
    manifest_file = open(manifest_path, "w", newline="", encoding="utf-8")
    manifest_writer = csv.writer(manifest_file)
    manifest_writer.writerow([
        "image_id", "filename", "source_group", "source_video", "source_frame",
        "timestamp_seconds", "augmentation", "dx_pixels", "dy_pixels",
        "rotation_deg", "scale", "instance_count"
    ])

    rng = random.Random(args.seed)
    image_id = 1
    annotation_id = 1
    video_stats = []

    print("\nPHASE C DATASET BUILDER V0")
    print(f"Raw root       : {raw_root}")
    print(f"Annotation root: {annotation_root}")
    print(f"Output         : {output_root}")
    print(f"regions.json   : {len(regions_files)} found")
    print(f"Class          : {CLASS_NAME}")

    for regions_json in regions_files:
        data = load_json(regions_json)
        filename = data.get("video", {}).get("filename")
        if not filename:
            print(f"Skipping {regions_json}: no video filename")
            continue

        video_path = find_raw_video(raw_root, filename)
        if video_path is None:
            print(f"Skipping {filename}: raw video not found")
            continue

        settings = get_settings(video_path.stem, config)
        required = [
            "augmentations_per_frame", "max_dx", "max_dy", "max_rotation_deg",
            "min_scale", "max_scale"
        ]
        missing = [key for key in required if key not in settings]
        if missing:
            raise ValueError(f"Missing settings for {video_path.stem}: {missing}")

        image_id, annotation_id, created = process_video(
            video_path,
            regions_json,
            images_dir,
            previews_dir,
            coco,
            manifest_writer,
            image_id,
            annotation_id,
            args.sample_every_seconds,
            settings,
            rng,
            args.preview_every,
        )
        video_stats.append({
            "video": video_path.name,
            "generated_images": created,
            "augmentation_settings": settings,
        })

    manifest_file.close()

    coco_path = annotations_dir / "instances_all.json"
    save_json(coco_path, coco)
    save_json(metadata_dir / "classes.json", {"1": CLASS_NAME})
    save_json(
        metadata_dir / "dataset_summary.json",
        {
            "dataset_version": "0.1",
            "class_strategy": "single-class instance segmentation",
            "class_name": CLASS_NAME,
            "sample_every_seconds": args.sample_every_seconds,
            "seed": args.seed,
            "total_images": len(coco["images"]),
            "total_instances": len(coco["annotations"]),
            "videos": video_stats,
            "note": (
                "V0 is an unsplit dataset pool. Train/val/test must later be split by source video/view, "
                "and all images sharing source_group must remain in the same split."
            ),
        },
    )

    print("\nDATASET BUILD COMPLETE")
    print(f"Images    : {len(coco['images'])}")
    print(f"Instances : {len(coco['annotations'])}")
    print(f"COCO JSON : {coco_path}")
    print(f"Manifest  : {manifest_path}")
    print(f"Previews  : {previews_dir}")
    print("\nNo train/val/test split was made yet. That is intentional with only a few real camera views.")


if __name__ == "__main__":
    main()
