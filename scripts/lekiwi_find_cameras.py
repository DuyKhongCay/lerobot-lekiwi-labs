#!/usr/bin/env python

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
Helper to find the camera devices available in your system, supporting both OpenCV, RealSense,
and Grayscale OpenCV cameras.

Example:

```shell
python pi5_labs/scripts/lekiwi_find_cameras.py
```
"""

import sys
from pathlib import Path

# Add project root and lerobot src to python path for importing
project_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(project_dir))

lerobot_src_dir = project_dir / "lerobot" / "src"
if lerobot_src_dir.exists():
    sys.path.append(str(lerobot_src_dir))

import argparse
import concurrent.futures
import logging
import time
import subprocess
import re
from typing import Any

import cv2  # type: ignore
import numpy as np
from PIL import Image

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.utils.errors import DeviceNotConnectedError

from pi5_labs.cameras.grayscale_opencv import GrayscaleCamOpenCV, GrayscaleCamOpenCVConfig

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Monkeypatch to resolve the race condition AttributeError in OpenCVCamera._read_loop
# -----------------------------------------------------------------------------
def safe_read_loop(self):
    """
    Background thread loop patched to avoid AttributeError: 'NoneType' object
    has no attribute 'is_set' when self.stop_event is set to None.
    """
    if self.stop_event is None:
        raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

    failure_count = 0
    stop_event = self.stop_event
    while not stop_event.is_set():
        try:
            raw_frame = self._read_from_hardware()
            processed_frame = self._postprocess_image(raw_frame)
            capture_time = time.perf_counter()

            with self.frame_lock:
                self.latest_frame = processed_frame
                self.latest_timestamp = capture_time
            self.new_frame_event.set()
            failure_count = 0

        except DeviceNotConnectedError:
            break
        except Exception as e:
            if failure_count <= 10:
                failure_count += 1
                logger.warning(f"Error reading frame in background thread for {self}: {e}")
            else:
                raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

# Apply the patch to OpenCVCamera
OpenCVCamera._read_loop = safe_read_loop
# -----------------------------------------------------------------------------


def get_real_video_devices() -> list[str] | None:
    """
    Uses v4l2-ctl --list-devices to parse and return only real video capture devices,
    excluding dummy platform ISP pipelines (like pispbe) and hardware codecs (hevc-dec).
    """
    try:
        result = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True, check=True)
        output = result.stdout
        
        devices = []
        current_group_is_ignored = False
        for line in output.splitlines():
            if not line.strip():
                continue
            if not line.startswith("\t") and not line.startswith(" "):
                group_name = line.lower()
                # Exclude platform ISP backends and codecs
                if any(k in group_name for k in ["pisp_be", "pispbe", "hevc-dec", "bcm2835", "rp1-cfe"]):
                    current_group_is_ignored = True
                else:
                    current_group_is_ignored = False
            else:
                if not current_group_is_ignored:
                    path = line.strip()
                    if path.startswith("/dev/video"):
                        devices.append(path)
        return devices
    except Exception as e:
        logger.warning(f"Failed to query v4l2-ctl: {e}. Falling back to default scanning.")
        return None


def find_all_opencv_cameras() -> list[dict[str, Any]]:
    """
    Finds all available OpenCV cameras plugged into the system,
    automatically classifying them as GrayscaleOpenCV or OpenCV.
    Excludes dummy platform ISP pipelines to prevent hangs.

    Returns:
        A list of all available OpenCV cameras with their metadata.
    """
    all_opencv_cameras_info: list[dict[str, Any]] = []
    logger.info("Searching for OpenCV cameras...")
    try:
        # Get real camera paths filtered by v4l2-ctl
        real_devices = get_real_video_devices()
        logger.info(f"v4l2-ctl filtered real capture devices: {real_devices}")
        
        opencv_cameras = OpenCVCamera.find_cameras()
        for cam_info in opencv_cameras:
            target = cam_info["id"]
            
            # If target is not in the filtered real capture devices list, skip it
            if real_devices is not None and str(target) not in real_devices:
                logger.debug(f"Skipping platform/ISP backend device: {target}")
                continue
            
            is_grayscale = False
            cam = None
            try:
                cfg = OpenCVCameraConfig(index_or_path=target, warmup_s=1)
                cam = OpenCVCamera(cfg)
                cam.connect(warmup=True)
                frame = cam.read()
                if frame is not None:
                    if frame.ndim == 2:
                        is_grayscale = True
                    elif frame.shape[2] == 1:
                        is_grayscale = True
                    elif frame.shape[2] == 3:
                        # OpenCVCamera.read() returns BGR or RGB depending on color_mode (default RGB).
                        # Let's check if the channels are identical
                        r = frame[:, :, 0]
                        g = frame[:, :, 1]
                        b = frame[:, :, 2]
                        if np.array_equal(r, g) and np.array_equal(g, b):
                            is_grayscale = True
            except Exception as e:
                logger.debug(f"Could not read frame from camera {target}: {e}")
            finally:
                if cam is not None:
                    try:
                        cam.disconnect()
                    except Exception:
                        pass

            if is_grayscale:
                cam_info["type"] = "GrayscaleOpenCV"
                cam_info["name"] = f"Grayscale OpenCV Camera @ {target}"
            else:
                cam_info["type"] = "OpenCV"
                cam_info["name"] = f"OpenCV Camera @ {target}"

            all_opencv_cameras_info.append(cam_info)
        logger.info(f"Found {len(all_opencv_cameras_info)} OpenCV/Grayscale cameras.")
    except Exception as e:
        logger.error(f"Error finding OpenCV cameras: {e}")

    return all_opencv_cameras_info


def find_all_realsense_cameras() -> list[dict[str, Any]]:
    """
    Finds all available RealSense cameras plugged into the system.

    Returns:
        A list of all available RealSense cameras with their metadata.
    """
    all_realsense_cameras_info: list[dict[str, Any]] = []
    logger.info("Searching for RealSense cameras...")
    try:
        realsense_cameras = RealSenseCamera.find_cameras()
        for cam_info in realsense_cameras:
            all_realsense_cameras_info.append(cam_info)
        logger.info(f"Found {len(realsense_cameras)} RealSense cameras.")
    except ImportError:
        logger.warning("Skipping RealSense camera search: pyrealsense2 library not found or not importable.")
    except Exception as e:
        logger.error(f"Error finding RealSense cameras: {e}")

    return all_realsense_cameras_info


def find_and_print_cameras(camera_type_filter: str | None = None) -> list[dict[str, Any]]:
    """
    Finds available cameras based on an optional filter and prints their information.

    Args:
        camera_type_filter: Optional string to filter cameras ("realsense", "opencv", or "grayscale_opencv").
                            If None, lists all cameras.

    Returns:
        A list of all available cameras matching the filter, with their metadata.
    """
    all_cameras_info: list[dict[str, Any]] = []

    if camera_type_filter:
        camera_type_filter = camera_type_filter.lower()

    if camera_type_filter is None or camera_type_filter in ["opencv", "grayscale_opencv"]:
        opencv_cams = find_all_opencv_cameras()
        if camera_type_filter == "opencv":
            opencv_cams = [c for c in opencv_cams if c["type"] == "OpenCV"]
        elif camera_type_filter == "grayscale_opencv":
            opencv_cams = [c for c in opencv_cams if c["type"] == "GrayscaleOpenCV"]
        all_cameras_info.extend(opencv_cams)
        
    if camera_type_filter is None or camera_type_filter == "realsense":
        all_cameras_info.extend(find_all_realsense_cameras())

    if not all_cameras_info:
        if camera_type_filter:
            logger.warning(f"No {camera_type_filter} cameras were detected.")
        else:
            logger.warning("No cameras (OpenCV, GrayscaleOpenCV, or RealSense) were detected.")
    else:
        print("\n--- Detected Cameras ---")
        for i, cam_info in enumerate(all_cameras_info):
            print(f"Camera #{i}:")
            for key, value in cam_info.items():
                if key == "default_stream_profile" and isinstance(value, dict):
                    print(f"  {key.replace('_', ' ').capitalize()}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key.capitalize()}: {sub_value}")
                else:
                    print(f"  {key.replace('_', ' ').capitalize()}: {value}")
            print("-" * 20)
    return all_cameras_info


def save_image(
    img_array: np.ndarray,
    camera_identifier: str | int,
    images_dir: Path,
    camera_type: str,
):
    """
    Saves a single image to disk using Pillow. Handles color conversion if necessary.
    """
    try:
        img = Image.fromarray(img_array, mode="RGB")

        safe_identifier = str(camera_identifier).replace("/", "_").replace("\\", "_")
        filename_prefix = f"{camera_type.lower()}_{safe_identifier}"
        filename = f"{filename_prefix}.png"

        path = images_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path))
        logger.info(f"Saved image: {path}")
    except Exception as e:
        logger.error(f"Failed to save image for camera {camera_identifier} (type {camera_type}): {e}")


def create_camera_instance(cam_meta: dict[str, Any]) -> dict[str, Any] | None:
    """Create and connect to a camera instance based on metadata."""
    cam_type = cam_meta.get("type")
    cam_id = cam_meta.get("id")
    instance = None

    logger.info(f"Preparing {cam_type} ID {cam_id} with default profile")

    try:
        if cam_type == "OpenCV":
            cv_config = OpenCVCameraConfig(
                index_or_path=cam_id,
                color_mode=ColorMode.RGB,
            )
            instance = OpenCVCamera(cv_config)
        elif cam_type == "GrayscaleOpenCV":
            cv_config = GrayscaleCamOpenCVConfig(
                index_or_path=cam_id,
                color_mode=ColorMode.RGB,
            )
            instance = GrayscaleCamOpenCV(cv_config)
        elif cam_type == "RealSense":
            rs_config = RealSenseCameraConfig(
                serial_number_or_name=cam_id,
                color_mode=ColorMode.RGB,
            )
            instance = RealSenseCamera(rs_config)
        else:
            logger.warning(f"Unknown camera type: {cam_type} for ID {cam_id}. Skipping.")
            return None

        if instance:
            logger.info(f"Connecting to {cam_type} camera: {cam_id}...")
            instance.connect(warmup=True)
            return {"instance": instance, "meta": cam_meta}
    except Exception as e:
        logger.error(f"Failed to connect or configure {cam_type} camera {cam_id}: {e}")
        if instance and instance.is_connected:
            instance.disconnect()
        return None


def process_camera_image(
    cam_dict: dict[str, Any], output_dir: Path, current_time: float
) -> concurrent.futures.Future | None:
    """Capture and process an image from a single camera."""
    cam = cam_dict["instance"]
    meta = cam_dict["meta"]
    cam_type_str = str(meta.get("type", "unknown"))
    cam_id_str = str(meta.get("id", "unknown"))

    try:
        image_data = cam.read()

        return save_image(
            image_data,
            cam_id_str,
            output_dir,
            cam_type_str,
        )
    except TimeoutError:
        logger.warning(
            f"Timeout reading from {cam_type_str} camera {cam_id_str} at time {current_time:.2f}s."
        )
    except Exception as e:
        logger.error(f"Error reading from {cam_type_str} camera {cam_id_str}: {e}")
    return None


def cleanup_cameras(cameras_to_use: list[dict[str, Any]]):
    """Disconnect all cameras."""
    logger.info(f"Disconnecting {len(cameras_to_use)} cameras...")
    for cam_dict in cameras_to_use:
        try:
            if cam_dict["instance"] and cam_dict["instance"].is_connected:
                cam_dict["instance"].disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting camera {cam_dict['meta'].get('id')}: {e}")


def save_images_from_all_cameras(
    output_dir: Path,
    record_time_s: float = 2.0,
    camera_type: str | None = None,
):
    """
    Connects to detected cameras (optionally filtered by type) and saves images from each.
    Uses default stream profiles for width, height, and FPS.

    Args:
        output_dir: Directory to save images.
        record_time_s: Duration in seconds to record images.
        camera_type: Optional string to filter cameras ("realsense", "opencv", or "grayscale_opencv").
                            If None, uses all detected cameras.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving images to {output_dir}")
    all_camera_metadata = find_and_print_cameras(camera_type_filter=camera_type)

    if not all_camera_metadata:
        logger.warning("No cameras detected matching the criteria. Cannot save images.")
        return

    cameras_to_use = []
    for cam_meta in all_camera_metadata:
        camera_instance = create_camera_instance(cam_meta)
        if camera_instance:
            cameras_to_use.append(camera_instance)

    if not cameras_to_use:
        logger.warning("No cameras could be connected. Aborting image save.")
        return

    logger.info(f"Starting image capture for {record_time_s} seconds from {len(cameras_to_use)} cameras.")
    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cameras_to_use) * 2) as executor:
        try:
            while time.perf_counter() - start_time < record_time_s:
                futures = []
                current_capture_time = time.perf_counter()

                for cam_dict in cameras_to_use:
                    future = process_camera_image(cam_dict, output_dir, current_capture_time)
                    if future:
                        futures.append(future)

                if futures:
                    concurrent.futures.wait(futures)

        except KeyboardInterrupt:
            logger.info("Capture interrupted by user.")
        finally:
            print("\nFinalizing image saving...")
            executor.shutdown(wait=True)
            cleanup_cameras(cameras_to_use)
            print(f"Image capture finished. Images saved to {output_dir}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(
        description="Unified camera utility script for listing cameras and capturing images."
    )

    parser.add_argument(
        "camera_type",
        type=str,
        nargs="?",
        default=None,
        choices=["realsense", "opencv", "grayscale_opencv"],
        help="Specify camera type to capture from (e.g., 'realsense', 'opencv', 'grayscale_opencv'). Captures from all if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs/captured_images",
        help="Directory to save images. Default: outputs/captured_images",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=6.0,
        help="Time duration to attempt capturing frames. Default: 6 seconds.",
    )
    args = parser.parse_args()
    save_images_from_all_cameras(**vars(args))


if __name__ == "__main__":
    main()
