#!/usr/bin/env python3

# Copyright 2026 LeKiwi Labs. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Wrapper for a single IMX219 Camera module connected to Raspberry Pi 5
via CSI port, using the Picamera2 library.
"""

import logging
import time
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation
from lerobot.cameras.utils import get_cv2_rotation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Picamera2 pixel format strings that correspond to OpenCV/numpy channel order
_FORMAT_MAP: dict[str, str] = {
    ColorMode.RGB: "RGB888",
    ColorMode.BGR: "BGR888",
}

# Nano-seconds per millisecond (SensorTimestamp is in nanoseconds)
_NS_PER_MS = 1_000_000

# Default sensor modes available on IMX219 (Width x Height @ max FPS)
_IMX219_MODES: list[dict[str, Any]] = [
    {"width": 640,  "height": 480,  "fps": 200},
    {"width": 1640, "height": 1232, "fps": 81},
    {"width": 1920, "height": 1080, "fps": 47},
    {"width": 3280, "height": 2464, "fps": 21},
]


@CameraConfig.register_subclass("imx219single")
@dataclass
class IMX219SingleCameraConfig(CameraConfig):
    """Configuration for a single IMX219 camera module on Raspberry Pi 5.

    Attributes:
        camera_idx: Index of the camera on the CSI connector (usually 0 or 1).
            Defaults to 0.
        color_mode: Output colour ordering (RGB or BGR). Defaults to RGB.
        rotation: Image rotation applied after capture. Defaults to NO_ROTATION.
        warmup_s: Seconds to wait after starting camera before returning
            from ``connect()``. This allows auto-exposure to settle.
            Defaults to 2.
        buffer_count: Number of frame buffers allocated. Defaults to 4.
        tuning_file: Path to custom tuning JSON file (optional).
    """
    camera_idx: int = 0
    color_mode: ColorMode = ColorMode.BGR
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 2
    buffer_count: int = 4
    tuning_file: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        if self.buffer_count < 4:
            raise ValueError("`buffer_count` must be at least 4.")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("`fps` must be a positive integer.")


class IMX219SingleCamera(Camera):
    """Single camera wrapper for a Sony IMX219 sensor on Raspberry Pi 5.

    Uses the Picamera2 library to capture frames independently without synchronization.
    """

    def __init__(self, config: IMX219SingleCameraConfig) -> None:
        """Initialise the single camera wrapper.

        Args:
            config: Configuration object for this camera.
        """
        super().__init__(config)
        self.config = config
        self.fps: int | None = config.fps
        self.width: int | None = config.width
        self.height: int | None = config.height

        self._picam = None
        self._frame_lock: Lock = Lock()
        self._new_frame_event: Event = Event()
        self._latest_frame: NDArray[Any] | None = None
        self._latest_timestamp: float | None = None
        self._started: bool = False

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        self.capture_width: int | None = None
        self.capture_height: int | None = None

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    @property
    def is_connected(self) -> bool:
        """True if camera is open and running."""
        return self._started

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Enumerate all cameras visible to libcamera / Picamera2."""
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as e:
            logger.warning(f"Picamera2 is not installed – cannot enumerate cameras. ({e})")
            return []

        try:
            raw_info = Picamera2.global_camera_info()
            cameras = []
            for item in raw_info:
                model = item.get("Model", "unknown")
                if "imx219" in model.lower():
                    cameras.append(
                        {
                            "index": item.get("Num", -1),
                            "model": model,
                            "location": item.get("Location", "unknown"),
                            "rotation": item.get("Rotation", 0),
                            "id": item.get("Id", ""),
                        }
                    )
            return cameras
        except Exception as e:
            logger.warning(f"Picamera2 is installed but failed to enumerate cameras: {e}")
            return []

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """Open the camera and start capturing frames."""
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise ConnectionError(
                "Picamera2 is not installed. Please run: sudo apt install python3-picamera2"
            ) from e

        logger.info(f"{self} opening camera (idx={self.config.camera_idx}) ...")
        try:
            self._picam = Picamera2(self.config.camera_idx, tuning=self.config.tuning_file)
        except Exception as e:
            raise ConnectionError(f"Failed to open camera: {e}") from e

        try:
            self._configure_capture_settings()
        except Exception as e:
            self._release_camera()
            raise ConnectionError(f"Failed to configure camera: {e}") from e

        self._picam.post_callback = self._callback

        try:
            self._picam.start()
            self._started = True
        except Exception as e:
            self._release_camera()
            raise ConnectionError(f"Failed to start camera: {e}") from e

        logger.info(f"{self} camera started. Output shape: ({self.height}, {self.width}, 3)")

        if warmup and self.config.warmup_s > 0:
            logger.info(f"{self} warming up for {self.config.warmup_s}s ...")
            deadline = time.time() + self.config.warmup_s
            while time.time() < deadline:
                self._new_frame_event.wait(timeout=0.1)
                self._new_frame_event.clear()

            with self._frame_lock:
                if self._latest_frame is None:
                    raise ConnectionError(
                        f"{self} warmup ended without receiving any frame. "
                        "Check that the camera is connected."
                    )
            logger.info(f"{self} warmup complete.")

    def _configure_capture_settings(self) -> None:
        """Applies the specified FPS, width, and height settings to the connected camera.

        This method configures the hardware to capture at the requested resolution (for full FOV),
        verifies the size applied by Picamera2, and falls back to 640x480 if necessary.
        """
        if self._picam is None:
            raise DeviceNotConnectedError(f"{self} picamera2 is not initialized")

        default_width = 640
        default_height = 480

        # Calculate capture resolution based on target width/height
        if self.width is None or self.height is None:
            self.width, self.height = default_width, default_height
            self.capture_width, self.capture_height = default_width, default_height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = default_height, default_width
                self.capture_width, self.capture_height = default_width, default_height
        else:
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width
            else:
                self.capture_width, self.capture_height = self.width, self.height

        if self.fps is None:
            self.fps = 30

        pixel_format = _FORMAT_MAP[self.config.color_mode]
        try:
            cfg = self._picam.create_preview_configuration(
                main={"format": pixel_format, "size": (self.capture_width, self.capture_height)},
                buffer_count=self.config.buffer_count,
                controls={"FrameRate": float(self.fps)},
            )
            self._picam.configure(cfg)
        except Exception as e:
            raise RuntimeError(f"Failed to configure picamera2: {e}")

        # Verify applied capture resolution from camera config
        actual_size = self._picam.camera_config["main"]["size"]
        actual_width, actual_height = actual_size[0], actual_size[1]

        # Fallback to 640x480 if hardware doesn't support requested capture resolution
        if self.capture_width != actual_width or self.capture_height != actual_height:
            logger.warning(
                f"[{self}] Failed to set hardware capture resolution to {self.capture_width}x{self.capture_height}. "
                f"Falling back to 640x480 capture on hardware."
            )
            try:
                cfg = self._picam.create_preview_configuration(
                    main={"format": pixel_format, "size": (640, 480)},
                    buffer_count=self.config.buffer_count,
                    controls={"FrameRate": float(self.fps)},
                )
                self._picam.configure(cfg)
            except Exception as e:
                raise RuntimeError(f"Failed to fallback to 640x480 configuration: {e}")

            actual_size = self._picam.camera_config["main"]["size"]
            actual_width, actual_height = actual_size[0], actual_size[1]

        self.capture_width = actual_width
        self.capture_height = actual_height

        # Determine target output width & height
        # Downsample to 640x480 (or 480x640 if rotated 90 deg) if capture resolution is larger than 640x480
        if self.capture_width is not None and self.capture_height is not None and (self.capture_width > 640 or self.capture_height > 480):
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = 480, 640
            else:
                self.width, self.height = 640, 480
        else:
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = self.capture_height, self.capture_width
            else:
                self.width, self.height = self.capture_width, self.capture_height

    def _resize_if_needed(self, image: NDArray[Any]) -> NDArray[Any]:
        """Resizes the processed image to the target output resolution (self.width, self.height) if they differ."""
        h, w = image.shape[:2]
        if self.width is not None and self.height is not None:
            if w != self.width or h != self.height:
                image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return image

    def _callback(self, request: Any) -> None:
        """Callback to handle newly captured frames."""
        try:
            frame: NDArray[Any] = request.make_array("main").copy()
        except Exception as e:
            logger.debug(f"{self} callback error: {e}")
            return

        frame = self._apply_rotation(frame)
        frame = self._resize_if_needed(frame)

        with self._frame_lock:
            self._latest_frame = frame
            self._latest_timestamp = time.perf_counter()
            self._new_frame_event.set()

    def _apply_rotation(self, frame: NDArray[Any]) -> NDArray[Any]:
        """Rotate frame using OpenCV rotation settings."""
        if self.rotation is not None and self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            return cv2.rotate(frame, self.rotation)
        return frame

    def read(self) -> NDArray[Any]:
        """Capture and return the next frame (blocking)."""
        start_time = time.perf_counter()
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._new_frame_event.clear()
        frame = self.async_read(timeout_ms=10_000)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Return the most recent new frame."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        arrived = self._new_frame_event.wait(timeout=timeout_ms / 1000.0)
        if not arrived:
            raise TimeoutError(f"{self} timed out waiting for a frame.")

        with self._frame_lock:
            frame = self._latest_frame
            self._new_frame_event.clear()

        if frame is None:
            raise RuntimeError("Internal error: event set but latest_frame is None.")
        return frame

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recently captured frame without blocking."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._frame_lock:
            frame = self._latest_frame
            timestamp = self._latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        logger.debug(f"{self} read_latest age: {age_ms:.1f}ms")
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self} latest frame is too stale: {age_ms:.1f} ms.")
        return frame

    def disconnect(self) -> None:
        """Stop camera and release all resources."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._release_camera()
        logger.info(f"{self} disconnected.")

    def _release_camera(self) -> None:
        """Clean up picamera2 instance and buffers."""
        if self._picam is not None:
            try:
                self._picam.post_callback = None
                if self._started:
                    self._picam.stop()
                self._picam.close()
            except Exception as e:
                logger.warning(f"Error releasing camera: {e}")
            finally:
                self._started = False
        self._picam = None

        with self._frame_lock:
            self._latest_frame = None
            self._latest_timestamp = None
            self._new_frame_event.clear()

    def __str__(self) -> str:
        return (
            f"IMX219SingleCamera("
            f"idx={self.config.camera_idx}, "
            f"{self.config.width}x{self.config.height}@{self.config.fps}fps)"
        )
