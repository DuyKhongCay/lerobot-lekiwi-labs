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


@CameraConfig.register_subclass("grayscale_opencv")
@dataclass
class GrayscaleOpenCVCamConfig(OpenCVCameraConfig):
    """Configuration class for grayscale-based OpenCV camera devices.
    
    Inherits all properties from OpenCVCameraConfig and registers under the choice "grayscale_opencv".
    """
    pass


class GrayscaleOpenCVCam(OpenCVCamera):
    """
    Manages camera interactions using OpenCV for capturing grayscale frames and
    converting them to RGB/BGR to be compatible with LeRobot's camera pipeline.
    
    Inherits all video capture, threading, and streaming logic from OpenCVCamera,
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
        success_exposure = self.videocapture.set(cv2.CAP_PROP_EXPOSURE, 650.0)
        if not success_exposure:
            logger.warning(f"{self} failed to set exposure to 650.")

        # Apply gain: 30
        success_gain = self.videocapture.set(cv2.CAP_PROP_GAIN, 30.0)
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

        return processed_image


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    """Creates OpenCV and GrayscaleCamOpenCV camera instances from configurations."""
    from typing import cast
    cameras: dict[str, Camera] = {}

    for key, cfg in camera_configs.items():
        if cfg.type == "opencv":
            from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
            from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
            cameras[key] = OpenCVCamera(cast(OpenCVCameraConfig, cfg))
        elif cfg.type == "grayscale_opencv":
            cameras[key] = GrayscaleOpenCVCam(cast(GrayscaleOpenCVCamConfig, cfg))
        else:
            raise ValueError(
                f"Unsupported camera type '{cfg.type}' for camera {key}. "
                f"Only 'opencv' and 'grayscale_opencv' are supported by this function."
            )

    return cameras


def duy0cay_find_cameras() -> list[dict[str, Any]]:
    # Override targets to scan on Linux to include lekiwi_* symlinks and deduplicate them
    if platform.system() == "Linux":
        dev_dir = Path("/dev")
        paths = list(dev_dir.glob("video*")) + list(dev_dir.glob("lekiwi_*"))
        
        resolved_to_paths = {}
        for p in paths:
            try:
                resolved = p.resolve()
                if resolved in resolved_to_paths:
                    # Prefer custom lekiwi_* symlink over raw video* paths
                    if p.name.startswith("lekiwi_") and not resolved_to_paths[resolved].name.startswith("lekiwi_"):
                        resolved_to_paths[resolved] = p
                else:
                    resolved_to_paths[resolved] = p
            except Exception:
                resolved_to_paths[p] = p
        
        possible_paths = sorted(list(resolved_to_paths.values()), key=lambda p: p.name)
        targets_to_scan = [str(p) for p in possible_paths]
    else:
        # Fallback to scanning standard indices on other systems
        targets_to_scan = [int(i) for i in range(60)]

    found_cameras_info = []
    for target in targets_to_scan:
        camera = cv2.VideoCapture(target)
        if camera.isOpened():
            default_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            default_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            default_fps = camera.get(cv2.CAP_PROP_FPS)
            default_format = camera.get(cv2.CAP_PROP_FORMAT)

            # Get FOURCC code and convert to string
            default_fourcc_code = camera.get(cv2.CAP_PROP_FOURCC)
            default_fourcc_code_int = int(default_fourcc_code)
            default_fourcc = "".join([chr((default_fourcc_code_int >> 8 * i) & 0xFF) for i in range(4)])

            camera_info = {
                "name": f"OpenCV Camera @ {target}",
                "type": "OpenCV",
                "id": target,
                "backend_api": camera.getBackendName(),
                "default_stream_profile": {
                    "format": default_format,
                    "fourcc": default_fourcc,
                    "width": default_width,
                    "height": default_height,
                    "fps": default_fps,
                },
            }

            found_cameras_info.append(camera_info)
            camera.release()

    return found_cameras_info


# Patch OpenCVCamera.find_cameras with our custom method
OpenCVCamera.find_cameras = staticmethod(duy0cay_find_cameras)


