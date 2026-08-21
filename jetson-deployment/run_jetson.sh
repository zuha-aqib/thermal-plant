#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE="${IMAGE:-engro-thermal-rtsp:rapidocr-arm64}"
PRESET="${PRESET:-cam_realmonitor}"

cd "$REPO_ROOT"

mkdir -p thermal-stream/presets
mkdir -p thermal-stream/screenshots

echo "============================================================"
echo "ENGRO THERMAL RTSP - JETSON CONTAINER"
echo "============================================================"
echo "Repo root : $REPO_ROOT"
echo "Image     : $IMAGE"
echo "Preset    : $PRESET"
echo "Display   : ${DISPLAY:-<not set>}"
echo
echo "The application will prompt for the RTSP URL / username / password."
echo "Camera credentials are not stored in this script."
echo

GUI_ARGS=()

if [ -n "${DISPLAY:-}" ] && [ -d /tmp/.X11-unix ]; then
    GUI_ARGS+=(
        -e "DISPLAY=$DISPLAY"
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    )
else
    echo "WARNING: DISPLAY/X11 was not detected."
    echo "The current application uses cv2.imshow/selectROI/mouse hover."
    echo "If the container connects but no window can open, configure X11 first."
    echo
fi

docker run --rm -it \
    --network host \
    --shm-size=512m \
    "${GUI_ARGS[@]}" \
    -v "$REPO_ROOT/thermal-stream/presets:/app/thermal-stream/presets" \
    -v "$REPO_ROOT/thermal-stream/screenshots:/app/thermal-stream/screenshots" \
    "$IMAGE" \
    --preset "$PRESET"
