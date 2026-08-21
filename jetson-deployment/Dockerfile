FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Runtime libraries needed by:
# - OpenCV HighGUI / imshow / selectROI
# - FFmpeg-backed RTSP decoding
# - OpenVINO CPU inference
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgtk-3-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# IMPORTANT:
# Build this Dockerfile from the THERMAL-PLANT REPO ROOT:
#
#   docker build \
#     -f jetson-deployment/Dockerfile \
#     -t engro-thermal-rtsp:rapidocr-arm64 \
#     .
#
# Therefore COPY paths below are relative to the repository root,
# NOT relative to jetson-deployment/.

COPY jetson-deployment/requirements-jetson.txt /app/requirements-jetson.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements-jetson.txt

# Existing project helpers remain at the repository root.
COPY step_code.py /app/step_code.py
COPY step_02_configure_thermal_scale.py /app/step_02_configure_thermal_scale.py
COPY step_03_process_thermal_temperatures.py /app/step_03_process_thermal_temperatures.py

# Deployment-specific live RTSP application.
RUN mkdir -p \
    /app/jetson-deployment \
    /app/thermal-stream/presets \
    /app/thermal-stream/screenshots

COPY jetson-deployment/thermal_rtsp_analysis.py \
     /app/jetson-deployment/thermal_rtsp_analysis.py

# Catch syntax/import-file mistakes during image creation.
RUN python -m py_compile \
    /app/step_code.py \
    /app/step_02_configure_thermal_scale.py \
    /app/step_03_process_thermal_temperatures.py \
    /app/jetson-deployment/thermal_rtsp_analysis.py

ENTRYPOINT ["python", "/app/jetson-deployment/thermal_rtsp_analysis.py"]
