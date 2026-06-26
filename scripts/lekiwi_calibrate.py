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
Custom calibration helper script for LeKiwi robot, integrating camera-skipping config
and monkeypatching OpenCVCamera to resolve connection race conditions.
"""

import sys
from pathlib import Path

# Add project root and lerobot src to python path for importing
project_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(project_dir))

lerobot_src_dir = project_dir / "lerobot" / "src"
if lerobot_src_dir.exists():
    sys.path.append(str(lerobot_src_dir))

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

# Import custom camera class for grayscale cameras
from lekiwi_labs.cameras.duy0cay_opencv import GrayscaleOpenCVCam

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)


from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense 
from lerobot.robots import (  # noqa: F401
    RobotConfig,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass
class CalibrateConfig:
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None
    skip_cameras: bool = True  # Default to True to bypass cameras for calibration

    def __post_init__(self):
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Choose either a teleop or a robot.")

        self.device = self.robot if self.robot else self.teleop


@draccus.wrap()
def calibrate(cfg: CalibrateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if isinstance(cfg.device, RobotConfig):
        device = make_robot_from_config(cfg.device)
    elif isinstance(cfg.device, TeleoperatorConfig):
        device = make_teleoperator_from_config(cfg.device)

    # Bypassing or handling camera connections gracefully
    if hasattr(device, "cameras"):
        if cfg.skip_cameras:
            logging.info("skip_cameras is True: bypassing camera initialization completely.")
            device.cameras = {}
        else:
            # Wrap camera connect methods to catch connection errors and log warnings
            for name, cam in list(device.cameras.items()):
                original_connect = cam.connect
                
                # Use a closure to capture scope correctly
                def make_safe_connect(camera_name, camera_obj, orig_conn):
                    def safe_connect(*args, **kwargs):
                        try:
                            orig_conn(*args, **kwargs)
                        except Exception as e:
                            logging.warning(
                                f"Failed to connect camera '{camera_name}' ({camera_obj}): {e}. "
                                "Continuing without this camera."
                            )
                            if hasattr(device, "cameras") and camera_name in device.cameras:
                                del device.cameras[camera_name]
                    return safe_connect
                
                cam.connect = make_safe_connect(name, cam, original_connect)

    device.connect(calibrate=False)

    try:
        device.calibrate()
    finally:
        device.disconnect()


def main():
    register_third_party_plugins()
    calibrate()


if __name__ == "__main__":
    main()
