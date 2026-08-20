#!/usr/bin/env python3
"""
live_camera_viewer.py

Low-latency RTSP camera viewer for testing a real network camera feed.

Features
--------
- RTSP input
- Secure password prompt (password is not stored in the script)
- TCP or UDP RTSP transport
- Background capture thread that continuously drains the stream
- Displays the newest available frame to reduce accumulated latency
- Automatic reconnect on stream failure
- Manual reconnect
- Freeze/unfreeze display while capture continues in the background
- Mouse hover with source-pixel coordinates and RGB value
- Screenshot saving
- Display FPS / stream status / resolution

This script intentionally does NOT contain thermal-temperature logic.
It is the live-camera foundation that can later be extended for a
radiometric/rendered thermal RTSP stream.

rtsp = rtsp://172.16.13.62:554/cam/realmonitor?channel=1&subtype=0
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlsplit, urlunsplit

import cv2
import numpy as np


WINDOW_NAME = "Live RTSP Camera"
DEFAULT_MAX_WIDTH = 1400
DEFAULT_MAX_HEIGHT = 850


def build_authenticated_rtsp_url(
    raw_url: str,
    username: Optional[str],
    password: Optional[str],
) -> str:
    """
    Insert username/password into an RTSP URL without printing them.

    If the URL already contains credentials, it is returned unchanged.
    """
    raw_url = raw_url.strip()

    if not raw_url.lower().startswith("rtsp://"):
        raise ValueError("The camera URL must start with rtsp://")

    parts = urlsplit(raw_url)

    if "@" in parts.netloc:
        return raw_url

    if not username:
        return raw_url

    user_enc = quote(username, safe="")
    pass_enc = quote(password or "", safe="")

    host_part = parts.netloc
    auth_netloc = f"{user_enc}:{pass_enc}@{host_part}"

    return urlunsplit(
        (parts.scheme, auth_netloc, parts.path, parts.query, parts.fragment)
    )


def redact_rtsp_url(url: str) -> str:
    """Return a safe URL string that never reveals credentials."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    except Exception:
        return "rtsp://<camera>"


class RTSPCapture:
    def __init__(
        self,
        url: str,
        transport: str = "tcp",
        reconnect_delay: float = 2.0,
    ) -> None:
        self.url = url
        self.transport = transport
        self.reconnect_delay = max(0.5, float(reconnect_delay))

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._force_reconnect_event = threading.Event()

        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_id = 0
        self.latest_frame_time = 0.0

        self.status = "IDLE"
        self.last_error = ""
        self.reconnect_count = 0

        self.source_width = 0
        self.source_height = 0
        self.source_fps = 0.0

        self._receive_times: deque[float] = deque(maxlen=180)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker,
            name="RTSPCaptureThread",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._force_reconnect_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        self._release_capture()

    def force_reconnect(self) -> None:
        self._force_reconnect_event.set()

    def _release_capture(self) -> None:
        cap = self._cap
        self._cap = None

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _configure_ffmpeg(self) -> None:
        # OpenCV's FFmpeg backend reads this environment variable.
        # TCP is usually more reliable for corporate / Wi-Fi RTSP networks.
        options = f"rtsp_transport;{self.transport}"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options

    def _open(self) -> bool:
        self._configure_ffmpeg()
        self.status = "CONNECTING"
        self.last_error = ""

        self._release_capture()

        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

            # Not every OpenCV backend honors this, but when supported it helps
            # avoid building a long queue of stale frames.
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                cap.release()
                self.last_error = "OpenCV could not open the RTSP stream."
                self.status = "RECONNECTING"
                return False

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

            self._cap = cap
            self.source_width = width
            self.source_height = height
            self.source_fps = fps if np.isfinite(fps) else 0.0

            self.status = "CONNECTED"
            return True

        except Exception as exc:
            self.last_error = str(exc)
            self.status = "RECONNECTING"
            self._release_capture()
            return False

    def _worker(self) -> None:
        first_connection = True

        while not self._stop_event.is_set():
            if not self._open():
                if not first_connection:
                    self.reconnect_count += 1
                first_connection = False

                self._stop_event.wait(self.reconnect_delay)
                continue

            first_connection = False
            consecutive_failures = 0

            while not self._stop_event.is_set():
                if self._force_reconnect_event.is_set():
                    self._force_reconnect_event.clear()
                    self.reconnect_count += 1
                    self.status = "RECONNECTING"
                    break

                cap = self._cap
                if cap is None:
                    break

                ok, frame = cap.read()

                if not ok or frame is None or frame.size == 0:
                    consecutive_failures += 1

                    if consecutive_failures >= 3:
                        self.last_error = "RTSP frame read failed."
                        self.status = "RECONNECTING"
                        self.reconnect_count += 1
                        break

                    time.sleep(0.03)
                    continue

                consecutive_failures = 0
                now = time.monotonic()

                with self._lock:
                    self.latest_frame = frame
                    self.latest_frame_id += 1
                    self.latest_frame_time = now
                    self._receive_times.append(now)

                self.status = "CONNECTED"

            self._release_capture()

            if not self._stop_event.is_set():
                self._stop_event.wait(self.reconnect_delay)

        self.status = "STOPPED"

    def get_latest(self) -> tuple[Optional[np.ndarray], int, float]:
        with self._lock:
            if self.latest_frame is None:
                return None, self.latest_frame_id, self.latest_frame_time

            return (
                self.latest_frame.copy(),
                self.latest_frame_id,
                self.latest_frame_time,
            )

    def measured_fps(self) -> float:
        with self._lock:
            times = list(self._receive_times)

        if len(times) < 2:
            return 0.0

        duration = times[-1] - times[0]
        if duration <= 0:
            return 0.0

        return (len(times) - 1) / duration


class LiveViewer:
    def __init__(
        self,
        capture: RTSPCapture,
        max_width: int,
        max_height: int,
        screenshot_dir: Path,
    ) -> None:
        self.capture = capture
        self.max_width = max_width
        self.max_height = max_height
        self.screenshot_dir = screenshot_dir

        self.frozen = False
        self.frozen_frame: Optional[np.ndarray] = None

        self.display_frame: Optional[np.ndarray] = None
        self.source_frame_for_hover: Optional[np.ndarray] = None

        self.display_scale = 1.0
        self.mouse_display_x = -1
        self.mouse_display_y = -1
        self.mouse_source_x = -1
        self.mouse_source_y = -1
        self.mouse_inside = False

        self.last_frame_id = -1

    def _fit_scale(self, width: int, height: int) -> float:
        if width <= 0 or height <= 0:
            return 1.0

        return min(
            1.0,
            self.max_width / width,
            self.max_height / height,
        )

    def _mouse_callback(self, event, x, y, flags, param) -> None:
        self.mouse_display_x = int(x)
        self.mouse_display_y = int(y)

        frame = self.source_frame_for_hover
        if frame is None:
            self.mouse_inside = False
            return

        h, w = frame.shape[:2]

        if self.display_scale <= 0:
            self.mouse_inside = False
            return

        src_x = int(round(x / self.display_scale))
        src_y = int(round(y / self.display_scale))

        if 0 <= src_x < w and 0 <= src_y < h:
            self.mouse_source_x = src_x
            self.mouse_source_y = src_y
            self.mouse_inside = True
        else:
            self.mouse_inside = False

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        scale: float = 0.55,
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_hover(self, display: np.ndarray) -> None:
        source = self.source_frame_for_hover

        if (
            source is None
            or not self.mouse_inside
            or self.mouse_source_x < 0
            or self.mouse_source_y < 0
        ):
            return

        x = self.mouse_source_x
        y = self.mouse_source_y

        b, g, r = [int(v) for v in source[y, x][:3]]

        dx = self.mouse_display_x
        dy = self.mouse_display_y

        cv2.drawMarker(
            display,
            (dx, dy),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

        lines = [
            f"Pixel: x={x}, y={y}",
            f"RGB: ({r}, {g}, {b})",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        thickness = 1
        padding = 8
        line_h = 22

        widths = []
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            widths.append(tw)

        box_w = max(widths) + padding * 2
        box_h = line_h * len(lines) + padding * 2

        box_x = dx + 16
        box_y = dy + 16

        h, w = display.shape[:2]

        if box_x + box_w >= w:
            box_x = max(0, dx - box_w - 16)

        if box_y + box_h >= h:
            box_y = max(0, dy - box_h - 16)

        overlay = display.copy()
        cv2.rectangle(
            overlay,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.72, display, 0.28, 0, display)

        for i, line in enumerate(lines):
            ty = box_y + padding + 16 + i * line_h
            cv2.putText(
                display,
                line,
                (box_x + padding, ty),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    def _draw_status(self, display: np.ndarray, frame_age: float) -> None:
        status = self.capture.status
        mode = "FROZEN" if self.frozen else "LIVE"

        fps = self.capture.measured_fps()
        source_fps = self.capture.source_fps

        h, w = display.shape[:2]

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 76), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.56, display, 0.44, 0, display)

        self._draw_text(
            display,
            f"{mode} | {status}",
            12,
            24,
            scale=0.62,
            thickness=1,
        )

        fps_text = f"Receive FPS: {fps:.1f}"
        if source_fps > 0.0:
            fps_text += f" | Camera FPS: {source_fps:.1f}"

        self._draw_text(
            display,
            fps_text,
            12,
            48,
            scale=0.50,
            thickness=1,
        )

        res_text = (
            f"Resolution: {self.capture.source_width}x{self.capture.source_height}"
            f" | Frame age: {frame_age * 1000:.0f} ms"
            f" | Reconnects: {self.capture.reconnect_count}"
        )
        self._draw_text(
            display,
            res_text,
            12,
            69,
            scale=0.46,
            thickness=1,
        )

        controls = "SPACE Freeze/Live   S Screenshot   R Reconnect   Q/ESC Quit"
        (tw, _), _ = cv2.getTextSize(
            controls,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            1,
        )

        overlay2 = display.copy()
        cv2.rectangle(
            overlay2,
            (0, h - 32),
            (w, h),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay2, 0.58, display, 0.42, 0, display)

        self._draw_text(
            display,
            controls,
            max(10, (w - tw) // 2),
            h - 10,
            scale=0.46,
            thickness=1,
        )

    def _waiting_frame(self, width: int = 960, height: int = 540) -> np.ndarray:
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        self._draw_text(
            frame,
            "Connecting to RTSP camera...",
            40,
            80,
            scale=0.9,
            thickness=2,
        )

        self._draw_text(
            frame,
            f"Status: {self.capture.status}",
            40,
            125,
            scale=0.65,
            thickness=1,
        )

        if self.capture.last_error:
            error = self.capture.last_error[:120]
            self._draw_text(
                frame,
                f"Last error: {error}",
                40,
                165,
                scale=0.5,
                thickness=1,
            )

        self._draw_text(
            frame,
            "The first RTSP connection can take several seconds.",
            40,
            215,
            scale=0.55,
            thickness=1,
        )

        self._draw_text(
            frame,
            "R = reconnect     Q/ESC = quit",
            40,
            260,
            scale=0.55,
            thickness=1,
        )

        return frame

    def toggle_freeze(self) -> None:
        if not self.frozen:
            latest, _, _ = self.capture.get_latest()
            if latest is not None:
                self.frozen_frame = latest
                self.frozen = True
        else:
            self.frozen = False
            self.frozen_frame = None

    def save_screenshot(self) -> Optional[Path]:
        source = self.source_frame_for_hover
        if source is None:
            return None

        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode = "frozen" if self.frozen else "live"

        path = self.screenshot_dir / f"rtsp_{mode}_{timestamp}.png"

        if cv2.imwrite(str(path), source):
            return path

        return None

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        self.capture.start()

        print()
        print("Live camera viewer started.")
        print("Controls:")
        print("  SPACE  Freeze / return to live")
        print("  S      Save screenshot")
        print("  R      Force RTSP reconnect")
        print("  Q/ESC  Quit")
        print()

        try:
            while True:
                latest, frame_id, frame_time = self.capture.get_latest()

                if self.frozen and self.frozen_frame is not None:
                    source = self.frozen_frame
                else:
                    source = latest

                if source is None:
                    display = self._waiting_frame()
                    self.source_frame_for_hover = None
                    self.display_scale = 1.0
                else:
                    self.source_frame_for_hover = source

                    h, w = source.shape[:2]
                    self.display_scale = self._fit_scale(w, h)

                    if self.display_scale != 1.0:
                        display_w = max(1, int(round(w * self.display_scale)))
                        display_h = max(1, int(round(h * self.display_scale)))

                        display = cv2.resize(
                            source,
                            (display_w, display_h),
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        display = source.copy()

                    frame_age = 0.0
                    if frame_time > 0:
                        frame_age = max(0.0, time.monotonic() - frame_time)

                    self._draw_status(display, frame_age)
                    self._draw_hover(display)

                self.display_frame = display
                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break

                if key == ord(" "):
                    self.toggle_freeze()

                elif key in (ord("r"), ord("R")):
                    print("Forcing RTSP reconnect...")
                    self.capture.force_reconnect()

                elif key in (ord("s"), ord("S")):
                    path = self.save_screenshot()
                    if path:
                        print(f"Screenshot saved: {path}")
                    else:
                        print("No frame available to save yet.")

                # If the OpenCV window was closed using the X button.
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break

        finally:
            self.capture.stop()
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-latency RTSP live camera viewer with mouse hover."
    )

    parser.add_argument(
        "--url",
        help="RTSP URL. If omitted, the script prompts for it.",
    )

    parser.add_argument(
        "--username",
        help="RTSP username. If omitted and the URL has no credentials, "
             "the script prompts for it.",
    )

    parser.add_argument(
        "--password",
        help="RTSP password. Avoid using this option on shared machines because "
             "command history may expose it. Prefer the secure prompt.",
    )

    parser.add_argument(
        "--transport",
        choices=("tcp", "udp"),
        default="tcp",
        help="RTSP transport. TCP is the default because it is usually more reliable.",
    )

    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=2.0,
        help="Seconds before reconnect attempts. Default: 2.0",
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=DEFAULT_MAX_WIDTH,
        help=f"Maximum display width. Default: {DEFAULT_MAX_WIDTH}",
    )

    parser.add_argument(
        "--max-height",
        type=int,
        default=DEFAULT_MAX_HEIGHT,
        help=f"Maximum display height. Default: {DEFAULT_MAX_HEIGHT}",
    )

    parser.add_argument(
        "--screenshot-dir",
        default="live-camera/screenshots",
        help="Directory for saved screenshots.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    raw_url = args.url
    if not raw_url:
        raw_url = input("RTSP URL: ").strip()

    if not raw_url:
        print("ERROR: No RTSP URL was provided.")
        return 2

    if not raw_url.lower().startswith("rtsp://"):
        print("ERROR: The URL must start with rtsp://")
        return 2

    parts = urlsplit(raw_url)
    url_has_credentials = "@" in parts.netloc

    username = args.username
    password = args.password

    if not url_has_credentials:
        if username is None:
            username = input("Username (leave blank if none): ").strip()

        if username and password is None:
            password = getpass.getpass("Password: ")

    try:
        final_url = build_authenticated_rtsp_url(raw_url, username, password)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    print()
    print("RTSP viewer configuration")
    print("-------------------------")
    print(f"Stream    : {redact_rtsp_url(final_url)}")
    print(f"Transport : {args.transport.upper()}")
    print("Password  : not displayed")
    print()

    capture = RTSPCapture(
        final_url,
        transport=args.transport,
        reconnect_delay=args.reconnect_delay,
    )

    viewer = LiveViewer(
        capture=capture,
        max_width=max(320, int(args.max_width)),
        max_height=max(240, int(args.max_height)),
        screenshot_dir=Path(args.screenshot_dir),
    )

    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
