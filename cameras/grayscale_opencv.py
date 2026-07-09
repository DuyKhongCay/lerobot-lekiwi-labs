# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Provides the GrayscaleCamOpenCV class for capturing frames from grayscale cameras using OpenCV
and converting them to RGB or BGR format for compatibility with the LeRobot pipeline.
"""

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import platform
import time
from typing import Any

from numpy.typing import NDArray  # type: ignore
import cv2  # type: ignore

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig, ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)


@CameraConfig.register_subclass("customopencv")
@dataclass
class CustomOpenCVCameraConfig(OpenCVCameraConfig):
    """Configuration class for custom OpenCV camera devices with auto-fallback to 640x480."""
    pass


class CustomOpenCVCamera(OpenCVCamera):
    """
    OpenCVCamera subclass that captures at configured resolution to get full FOV,
    but automatically downsamples the output resolution to 640x480 (or 480x640 if rotated)
    instead of cropping, ensuring compatibility with the data pipeline.
    """

    def _validate_width_and_height(self) -> None:
        """Validates and sets the camera's frame capture width and height.
        Configures the hardware to capture at the requested resolution (for full FOV),
        but downsamples the output resolution to 640x480.
        """
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        if self.capture_width is None or self.capture_height is None:
            raise ValueError(f"{self} capture_width or capture_height is not set")

        # Try to set the requested capture resolution on hardware
        width_success = self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.capture_width))
        height_success = self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.capture_height))

        actual_width = round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # If setting the capture resolution failed, fallback to 640x480 capture on hardware
        if not width_success or self.capture_width != actual_width or not height_success or self.capture_height != actual_height:
            logger.warning(
                f"[{self}] Failed to set hardware capture resolution to {self.capture_width}x{self.capture_height}. "
                f"Falling back to 640x480 capture on hardware."
            )
            self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, 640.0)
            self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480.0)
            actual_width = round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Update capture properties to reflect actual hardware settings
        self.capture_width = actual_width
        self.capture_height = actual_height

        # Force target output resolution to 640x480 (swapped to 480x640 if rotated 90 deg)
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.width, self.height = 480, 640
        else:
            self.width, self.height = 640, 480

    def _resize_if_needed(self, image: NDArray[Any]) -> NDArray[Any]:
        """Resizes the processed image to the target output resolution (self.width, self.height) if they differ."""
        h, w = image.shape[:2]
        if self.width is not None and self.height is not None:
            if w != self.width or h != self.height:
                image = cv2.resize(image, (int(self.width), int(self.height)), interpolation=cv2.INTER_AREA)
        return image

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        """Applies OpenCVCamera postprocessing, then downsamples to the target output size."""
        processed = super()._postprocess_image(image)
        return self._resize_if_needed(processed)


@CameraConfig.register_subclass("grayscaleopencv")
@dataclass
class GrayscaleOpenCVCamConfig(CustomOpenCVCameraConfig):
    """Configuration class for grayscale-based OpenCV camera devices.
    
    Inherits all properties from CustomOpenCVCameraConfig and registers under the choice "grayscale_opencv".
    """
    pass


class GrayscaleOpenCVCam(CustomOpenCVCamera):
    """
    Manages camera interactions using OpenCV for capturing grayscale frames and
    converting them to RGB/BGR to be compatible with LeRobot's camera pipeline.
    
    Inherits all video capture, threading, and streaming logic from CustomOpenCVCamera,
    overriding only the postprocessing logic.
    """

    @check_if_not_connected
    def _configure_capture_settings(self) -> None:
        """
        Applies standard camera settings, then configures exposure to 650 and gain to 30.
        """
        super()._configure_capture_settings()

        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        # Grab a dummy frame and sleep to trigger stream start so driver doesn't override controls
        self.videocapture.grab()
        time.sleep(0.2)

        # Disable auto exposure (set to manual)
        success_auto = self.videocapture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        if not success_auto:
            logger.warning(f"{self} failed to set auto exposure to manual (1).")

        # Apply exposure: 650 (on a scale of 10000)
        success_exposure = self.videocapture.set(cv2.CAP_PROP_EXPOSURE, 400.0)
        if not success_exposure:
            logger.warning(f"{self} failed to set exposure to 650.")

        # Apply gain: 30
        success_gain = self.videocapture.set(cv2.CAP_PROP_GAIN, 20)
        if not success_gain:
            logger.warning(f"{self} failed to set gain to 30.")

        actual_auto = self.videocapture.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        actual_exposure = self.videocapture.get(cv2.CAP_PROP_EXPOSURE)
        actual_gain = self.videocapture.get(cv2.CAP_PROP_GAIN)
        logger.warning(f"[{self}] CONFIG DETAILS -> auto_exposure set success: {success_auto} (actual: {actual_auto}), exposure set success: {success_exposure} (actual: {actual_exposure}), gain set success: {success_gain} (actual: {actual_gain})")

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        """
        Applies grayscale-to-color conversion, dimension validation, and rotation to a raw frame.

        Args:
            image (np.ndarray): The raw image frame (expected to be grayscale or BGR).

        Returns:
            np.ndarray: The processed 3-channel (RGB or BGR) image frame.
        """
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        # Parse shape and extract single-channel grayscale image
        if len(image.shape) == 2:
            h, w = image.shape
            c = 1
            gray_image = image
        elif len(image.shape) == 3:
            h, w, c = image.shape
            if c == 1:
                gray_image = image[:, :, 0]
            elif c == 3:
                # If OpenCV decoded it as a 3-channel BGR, convert it to grayscale first
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                raise RuntimeError(f"{self} frame channels={c} is not supported. Expected 1 or 3 channels.")
        else:
            raise RuntimeError(f"{self} frame dimensions={image.ndim} is not supported. Expected 2 or 3 dims.")

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        # Convert grayscale image to the configured 3-channel color mode
        if self.color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        else:
            processed_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        # Apply rotation if configured
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        # Resize if the capture resolution is larger than target output resolution
        processed_image = self._resize_if_needed(processed_image)

        return processed_image

