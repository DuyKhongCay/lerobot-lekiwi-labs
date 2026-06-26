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
Custom host script for LeKiwi robot, overriding LeKiwiConfig and LeKiwiClientConfig
to use a custom lekiwi_cameras_config function with custom device configurations.
"""

import base64
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import draccus
import zmq

# Add project root and lerobot src to python path for importing
project_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(project_dir))

lerobot_src_dir = project_dir / "lerobot" / "src"
if lerobot_src_dir.exists():
    sys.path.append(str(lerobot_src_dir))

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.lekiwi.config_lekiwi import (
    LeKiwiConfig as OriginalLeKiwiConfig,
    LeKiwiClientConfig as OriginalLeKiwiClientConfig,
    LeKiwiHostConfig,
)
from lerobot.robots.lekiwi.lekiwi import LeKiwi as OriginalLeKiwi

# Import custom Grayscale camera configuration from either lekiwi_labs or pi5_labs
from lekiwi_labs.cameras.duy0cay_opencv import GrayscaleOpenCVCamConfig, make_cameras_from_configs as make_duy0cay_cameras


def lekiwi_cameras_config() -> dict[str, CameraConfig]:
    """
    Custom camera configuration function for LeKiwi robot.
    """
    return {
        # "front": OpenCVCameraConfig(
        #     index_or_path="/dev/lekiwi_front_steoreocam",
        #     fps=30,
        #     width=640,
        #     height=480,
        #     rotation=Cv2Rotation.ROTATE_180,
        # ),
        "wrist": GrayscaleOpenCVCamConfig(
            index_or_path="/dev/lekiwi_wrist_cam",
            fps=30,
            width=480,
            height=640,
            rotation=Cv2Rotation.ROTATE_90,
        ),
        "side": OpenCVCameraConfig(
            index_or_path="/dev/lekiwi_side_webcam",
            fps=30,
            width=480,
            height=640,
            rotation=Cv2Rotation.ROTATE_90,
        ),
    }


@RobotConfig.register_subclass("duy0cay_lekiwi")
@dataclass(kw_only=True)
class LeKiwiConfig(OriginalLeKiwiConfig):
    # Set default ID to load DuyKhongCay calibration by default
    id: str | None = "DuyKhongCay"
    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)


@RobotConfig.register_subclass("duy0cay_lekiwi_client")
@dataclass(kw_only=True)
class LeKiwiClientConfig(OriginalLeKiwiClientConfig):
    # Set default ID to load DuyKhongCay calibration by default
    id: str | None = "DuyKhongCay"
    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)


@dataclass
class LeKiwiServerConfig:
    """Configuration for the LeKiwi host script."""

    robot: LeKiwiConfig = field(default_factory=LeKiwiConfig)
    host: LeKiwiHostConfig = field(default_factory=LeKiwiHostConfig)


class LeKiwi(OriginalLeKiwi):
    config_class = LeKiwiConfig

    def __init__(self, config: LeKiwiConfig):
        # Temporarily patch make_cameras_from_configs in the original module
        # to use our custom make_duy0cay_cameras factory.
        import lerobot.robots.lekiwi.lekiwi as lekiwi_module
        original_make_cameras = lekiwi_module.make_cameras_from_configs
        lekiwi_module.make_cameras_from_configs = make_duy0cay_cameras
        try:
            super().__init__(config)
        finally:
            lekiwi_module.make_cameras_from_configs = original_make_cameras

    def calibrate(self) -> None:
        # Overridden to automatically load and write the calibration file
        # without blocking for interactive input on the host.
        if self.calibration:
            logging.info(f"Automatically writing calibration file associated with the id {self.id} to the motors")
            self.bus.write_calibration(self.calibration)
        else:
            logging.warning(f"No calibration file found for id {self.id}. Proceeding to interactive calibration.")
            super().calibrate()


class LeKiwiHost:
    def __init__(self, config: LeKiwiHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


@draccus.wrap()
def main(cfg: LeKiwiServerConfig):
    logging.info("Configuring LeKiwi")
    robot = LeKiwi(cfg.robot)

    logging.info("Connecting LeKiwi")
    robot.connect()

    logging.info("Starting HostAgent")
    host = LeKiwiHost(cfg.host)

    last_cmd_time = time.time()
    watchdog_active = False
    logging.info("Waiting for commands...")
    try:
        # Business logic
        start = time.perf_counter()
        duration = 0
        while duration < host.connection_time_s:
            loop_start_time = time.time()
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                _action_sent = robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                if not watchdog_active:
                    logging.warning("No command available")
            except Exception as e:
                logging.error("Message fetching failed: %s", e)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping the base."
                )
                watchdog_active = True
                robot.stop_base()

            last_observation = robot.get_observation()

            # Encode ndarrays to base64 strings
            for cam_key, _ in robot.cameras.items():
                ret, buffer = cv2.imencode(
                    ".jpg", last_observation[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                )
                if ret:
                    last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
                else:
                    last_observation[cam_key] = ""

            # Send the observation to the remote agent
            try:
                host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.info("Dropping observation, no client connected")

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.time() - loop_start_time

            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start
        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down Lekiwi Host.")
        robot.disconnect()
        host.disconnect()

    logging.info("Finished LeKiwi cleanly")


if __name__ == "__main__":
    main()
