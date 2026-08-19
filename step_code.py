"""Shared provenance/state helpers for the Engro thermal-video pipeline.

The goal of this module is simple: an output is not considered current just
because a file exists.  It is current only when the exact inputs and settings
that produced it still match the inputs/settings requested now.

All stage scripts and synthetic-video generators use this file so completion,
staleness, hashing, prompts, and safe replacement behave consistently.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


MANIFEST_VERSION = 1
HASH_CHUNK_SIZE = 8 * 1024 * 1024


# ============================================================
# GENERIC FILE / JSON HELPERS
# ============================================================


def now_utc_iso() -> str:
    """Return a stable ISO-8601 UTC timestamp for manifests."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_exists_and_nonempty(path: Union[Path, str]) -> bool:
    """True only when *path* is a regular non-empty file."""
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def has_any_output(path: Union[Path, str]) -> bool:
    """True when a directory exists and contains at least one item."""
    path = Path(path)

    if not path.is_dir():
        return False

    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False


def safe_remove(path: Union[Path, str]) -> None:
    """Remove a file or directory if it exists."""
    path = Path(path)

    if not path.exists():
        return

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def prompt_yes_no(message: str, default: bool = True) -> bool:
    """Reusable interactive Y/N prompt."""
    suffix = " [Y/n]: " if default else " [y/N]: "

    while True:
        answer = input(message + suffix).strip().lower()

        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False

        print("Please enter Y or N.")


def load_json(path: Union[Path, str], default: Any = None) -> Any:
    """Read JSON; return *default* when the file does not exist."""
    path = Path(path)

    if not path.is_file():
        return default

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json_atomic(path: Union[Path, str], data: Any) -> None:
    """Write JSON atomically so interruption cannot leave half a manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_name(path.name + ".__writing__")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
        file.write("\n")

    os.replace(temp_path, path)


# ============================================================
# HASHING / FINGERPRINTING
# ============================================================


def sha256_file(path: Union[Path, str]) -> str:
    """Hash the complete file contents with SHA-256."""
    path = Path(path)
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def canonical_json_hash(data: Any) -> str:
    """Hash JSON meaning, ignoring whitespace/key-order formatting changes."""
    encoded = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def hash_json_file(path: Union[Path, str]) -> str:
    """Canonical SHA-256 of a JSON file."""
    path = Path(path)

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return canonical_json_hash(data)


def fingerprint_file(path: Union[Path, str], *, content_hash: bool = True) -> Dict[str, Any]:
    """Return provenance metadata for an arbitrary file."""
    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    stat = path.stat()

    result: Dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }

    if content_hash:
        result["sha256"] = sha256_file(path)

    return result


def fingerprint_video(path: Union[Path, str]) -> Dict[str, Any]:
    """Fingerprint the full raw-video bytes.

    A complete SHA-256 is intentionally used.  Therefore a replacement MP4
    with the same filename is still detected as a changed dependency.
    """
    return fingerprint_file(path, content_hash=True)


def same_file_content(old: Dict[str, Any], new: Dict[str, Any]) -> bool:
    """Compare two fingerprints by content hash when available."""
    if not old or not new:
        return False

    if old.get("sha256") and new.get("sha256"):
        return old["sha256"] == new["sha256"]

    return (
        old.get("size_bytes") == new.get("size_bytes")
        and old.get("mtime_ns") == new.get("mtime_ns")
    )


# ============================================================
# MANIFEST HELPERS
# ============================================================


def make_manifest(
    stage: str,
    *,
    source_video: Optional[Dict[str, Any]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    settings: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a standard completed-run manifest."""
    manifest: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "stage": stage,
        "status": "completed",
        "created_at_utc": now_utc_iso(),
        "source_video": source_video or {},
        "inputs": inputs or {},
        "settings": settings or {},
        "outputs": outputs or {},
    }

    if extra:
        manifest.update(extra)

    return manifest


def load_manifest(path: Union[Path, str]) -> Optional[Dict[str, Any]]:
    """Return a manifest dictionary, or None if unavailable/unreadable."""
    path = Path(path)

    if not path.is_file():
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    return data


def manifest_signature(data: Any) -> str:
    """Short stable ID useful for synthetic parameter variants."""
    return canonical_json_hash(data)[:16]


def compare_manifest_sections(
    existing: Dict[str, Any],
    expected: Dict[str, Any],
    checks: Iterable[Tuple[str, str, Optional[str]]],
) -> List[str]:
    """Compare selected manifest values and return human-readable reasons.

    checks contains tuples:
        (section, key, reason_if_different)

    A special section name ``root`` reads directly from the manifest root.
    """
    reasons: List[str] = []

    for section, key, reason in checks:
        if section == "root":
            old_value = existing.get(key)
            new_value = expected.get(key)
        else:
            old_value = existing.get(section, {}).get(key)
            new_value = expected.get(section, {}).get(key)

        if old_value != new_value:
            reasons.append(reason or f"{section}.{key} changed")

    return reasons


def verify_required_files(output_dir: Union[Path, str], filenames: Iterable[str]) -> bool:
    """Check that every named required output exists and is non-empty."""
    output_dir = Path(output_dir)
    return all(file_exists_and_nonempty(output_dir / name) for name in filenames)


# ============================================================
# SAFE DIRECTORY / FILE REPLACEMENT
# ============================================================


def promote_completed_directory(working_dir: Union[Path, str], final_dir: Union[Path, str]) -> None:
    """Atomically-ish replace a completed output directory.

    The old completed result is kept as a short-lived sibling backup while the
    new working directory is promoted.  If promotion fails, the old result is
    restored.
    """
    working_dir = Path(working_dir)
    final_dir = Path(final_dir)

    final_dir.parent.mkdir(parents=True, exist_ok=True)

    backup_dir = final_dir.with_name(final_dir.name + ".__backup__")
    safe_remove(backup_dir)

    if final_dir.exists():
        final_dir.rename(backup_dir)

    try:
        working_dir.rename(final_dir)
    except Exception:
        if backup_dir.exists() and not final_dir.exists():
            backup_dir.rename(final_dir)
        raise
    else:
        safe_remove(backup_dir)


def replace_completed_file(temp_path: Union[Path, str], final_path: Union[Path, str]) -> None:
    """Replace one completed file only after its temporary file exists."""
    temp_path = Path(temp_path)
    final_path = Path(final_path)

    if not file_exists_and_nonempty(temp_path):
        raise RuntimeError(f"Temporary output is missing/empty: {temp_path}")

    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_path, final_path)
