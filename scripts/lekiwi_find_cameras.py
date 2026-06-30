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
and Grayscale OpenCV cameras. This script refactors the custom find-cameras logic by import
and monkey-patching the original lerobot_find_cameras.py script.

Example:

```shell
python lekiwi_labs/scripts/lekiwi_find_cameras.py
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
import logging
import subprocess
from typing import Any

import numpy as np

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

# Import custom camera configuration
from lekiwi_labs.cameras.grayscale_opencv import GrayscaleOpenCVCamConfig, GrayscaleOpenCVCam

# Import the original lerobot script to monkey-patch it
import lerobot.scripts.lerobot_find_cameras as lfc

logger = logging.getLogger(__name__)


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
    logger.info("Searching for OpenCV cameras (patched with lekiwi filters)...")
    try:
        # Get real camera paths filtered by v4l2-ctl
        real_devices = get_real_video_devices()
        logger.info(f"v4l2-ctl filtered real capture devices: {real_devices}")
        
        opencv_cameras = OpenCVCamera.find_cameras()
        for cam_info in opencv_cameras:
            target = cam_info["id"]
            
            # If target is not in the filtered real capture devices list (check both original and resolved path), skip it
            resolved_target = str(Path(target).resolve())
            if real_devices is not None and str(target) not in real_devices and resolved_target not in real_devices:
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


def create_camera_instance(cam_meta: dict[str, Any]) -> dict[str, Any] | None:
    """Create and connect to a camera instance based on metadata (patched for GrayscaleOpenCV support)."""
    cam_type = cam_meta.get("type")
    cam_id = cam_meta.get("id")
    instance = None

    logger.info(f"Preparing {cam_type} ID {cam_id} with default profile (patched)")

    try:
        if cam_type == "OpenCV":
            cv_config = OpenCVCameraConfig(
                index_or_path=cam_id,
                color_mode=ColorMode.RGB,
            )
            instance = OpenCVCamera(cv_config)
        elif cam_type == "GrayscaleOpenCV":
            cv_config = GrayscaleOpenCVCamConfig(
                index_or_path=cam_id,
                color_mode=ColorMode.RGB,
            )
            instance = GrayscaleOpenCVCam(cv_config)
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


def find_and_print_cameras(camera_type_filter: str | None = None) -> list[dict[str, Any]]:
    """
    Finds available cameras based on an optional filter and prints their information (patched).
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
        all_cameras_info.extend(lfc.find_all_realsense_cameras())

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


# Apply Monkey Patches to the original lerobot script
lfc.find_all_opencv_cameras = find_all_opencv_cameras
lfc.create_camera_instance = create_camera_instance
lfc.find_and_print_cameras = find_and_print_cameras


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
    
    # Run the original save function which now uses our patched callbacks
    lfc.save_images_from_all_cameras(**vars(args))


if __name__ == "__main__":
    main()
