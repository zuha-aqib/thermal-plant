#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "JETSON DEPLOYMENT CHECK"
echo "============================================================"

echo
echo "[Repository]"
echo "Repo root: $REPO_ROOT"

required_files=(
    "$REPO_ROOT/jetson-deployment/Dockerfile"
    "$REPO_ROOT/jetson-deployment/Dockerfile.dockerignore"
    "$REPO_ROOT/jetson-deployment/requirements-jetson.txt"
    "$REPO_ROOT/jetson-deployment/thermal_rtsp_analysis.py"
    "$REPO_ROOT/step_code.py"
    "$REPO_ROOT/step_02_configure_thermal_scale.py"
    "$REPO_ROOT/step_03_process_thermal_temperatures.py"
)

missing=0

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "OK      $file"
    else
        echo "MISSING $file"
        missing=1
    fi
done

echo
echo "[Host]"
echo "Architecture: $(uname -m)"
echo "Kernel      : $(uname -r)"

if command -v docker >/dev/null 2>&1; then
    echo "Docker      : $(docker --version)"
else
    echo "Docker      : NOT FOUND"
    missing=1
fi

if command -v tegrastats >/dev/null 2>&1; then
    echo "tegrastats  : $(command -v tegrastats)"
else
    echo "tegrastats  : not found"
fi

echo
echo "[Docker runtimes]"
docker info --format '{{json .Runtimes}}' 2>/dev/null || true

echo
echo "[GUI]"
echo "DISPLAY     : ${DISPLAY:-<not set>}"

if [ -d /tmp/.X11-unix ]; then
    echo "X11 socket  : present"
else
    echo "X11 socket  : not present"
fi

echo
if [ "$(uname -m)" != "aarch64" ]; then
    echo "WARNING: Jetson is expected to report aarch64."
    missing=1
fi

if [ "$missing" -ne 0 ]; then
    echo "One or more deployment checks need attention."
    exit 1
fi

echo "Deployment prerequisites look good."
