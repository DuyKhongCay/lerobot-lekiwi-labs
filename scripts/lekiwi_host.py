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
import numpy as np
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
    LeKiwiConfig,
    LeKiwiClientConfig,
    LeKiwiHostConfig,
)
from lerobot.robots.lekiwi.lekiwi import LeKiwi as OriginalLeKiwi
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient as OriginalLeKiwiClient
from lerobot.robots.lekiwi.lekiwi_host import LeKiwiHost, LeKiwiServerConfig
from lerobot.types import RobotAction

# Import custom Grayscale camera configuration from either lekiwi_labs or pi5_labs
from lekiwi_labs.cameras.duy0cay_opencv import make_cameras_from_configs as duy0cay_make_cameras


class LeKiwi(OriginalLeKiwi):
    config_class = LeKiwiConfig

    def __init__(self, config: LeKiwiConfig):
        # Temporarily patch make_cameras_from_configs in the original module
        # to use our custom make_duy0cay_cameras factory.
        import lerobot.robots.lekiwi.lekiwi as lekiwi_module
        original_make_cameras = lekiwi_module.make_cameras_from_configs
        lekiwi_module.make_cameras_from_configs = duy0cay_make_cameras
        try:
            super().__init__(config)
        finally:
            lekiwi_module.make_cameras_from_configs = original_make_cameras

        # Initialize internal keyboard teleop for controlling the base
        from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
        self.keyboard_teleop = KeyboardTeleop(KeyboardTeleopConfig(id="lekiwi_internal_keyboard"))

        # Speed control and keyboard keys setup for local LeKiwi robot base
        self.teleop_keys = {
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            "speed_up": "r",
            "speed_down": "f",
            "quit": "q",
        }
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # slow
            {"xy": 0.2, "theta": 60},  # medium
            {"xy": 0.3, "theta": 90},  # fast
        ]
        self.speed_index = 0

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        self.keyboard_teleop.connect()

    def disconnect(self) -> None:
        if hasattr(self, "keyboard_teleop"):
            self.keyboard_teleop.disconnect()
        super().disconnect()

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray) -> dict[str, float]:
        # Speed control
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, 2)
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]
        theta_speed = speed_setting["theta"]

        x_cmd = 0.0
        y_cmd = 0.0
        theta_cmd = 0.0

        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd += xy_speed
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd -= xy_speed
        if self.teleop_keys["left"] in pressed_keys:
            y_cmd += xy_speed
        if self.teleop_keys["right"] in pressed_keys:
            y_cmd -= xy_speed
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += theta_speed
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= theta_speed
        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
        }

    def send_action(self, action: RobotAction) -> RobotAction:
        if self.keyboard_teleop.is_connected:
            pressed_keys = self.keyboard_teleop.get_action()
            keys_list = list(pressed_keys.keys())
            base_action = self._from_keyboard_to_base_action(np.array(keys_list))
            # Merge base action into the action command
            action = {**action, **base_action}
        return super().send_action(action)

    def calibrate(self) -> None:
        # Overridden to automatically load and write the calibration file
        # without blocking for interactive input on the host.
        if self.calibration:
            logging.info(f"Automatically writing calibration file associated with the id {self.id} to the motors")
            self.bus.write_calibration(self.calibration)
        else:
            logging.warning(f"No calibration file found for id {self.id}. Proceeding to interactive calibration.")
            super().calibrate()


class LeKiwiClient(OriginalLeKiwiClient):
    config_class = LeKiwiClientConfig

    def __init__(self, config: LeKiwiClientConfig):
        super().__init__(config)
        # Initialize internal keyboard teleop for controlling the base
        from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
        self.keyboard_teleop = KeyboardTeleop(KeyboardTeleopConfig(id="lekiwi_internal_keyboard"))

    def connect(self) -> None:
        super().connect()
        self.keyboard_teleop.connect()

    def disconnect(self) -> None:
        if hasattr(self, "keyboard_teleop"):
            self.keyboard_teleop.disconnect()
        super().disconnect()

    def send_action(self, action: RobotAction) -> RobotAction:
        if self.keyboard_teleop.is_connected:
            pressed_keys = self.keyboard_teleop.get_action()
            keys_list = list(pressed_keys.keys())
            base_action = self._from_keyboard_to_base_action(np.array(keys_list))
            # Merge base action into the action command before forwarding to host
            action = {**action, **base_action}
        return super().send_action(action)



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
