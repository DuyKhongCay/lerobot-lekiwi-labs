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
Specialized script to control the LeKiwi robot from teleoperation (leader arm + base controller).

Example with uArm leader and laptop keyboard base control:

```shell
python lekiwi_labs/scripts/lekiwi_teleoperate.py \
    --robot.type=lekiwi \
    --robot.port=/dev/ttyACM0 \
    --teleop.type=uarm_leader_config2 \
    --teleop.port=/dev/ttyUSB0 \
    --base_teleop.type=keyboard \
    --display_data=true
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

# Add uarm-leader-config1 to python path so its modules can be found
uarm_config_dir = project_dir / "lekiwi_labs" / "teleoperates" / "uarm-leader-config1"
if uarm_config_dir.exists():
    sys.path.append(str(uarm_config_dir))

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any

import rerun as rr

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)


# Import camera configuration classes for side-effect registration
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401

# Import custom grayscale camera configuration
try:
    from lekiwi_labs.cameras.grayscale_opencv import GrayscaleCamOpenCVConfig  # noqa: F401
except ImportError:
    from pi5_labs.cameras.grayscale_opencv import GrayscaleCamOpenCVConfig  # noqa: F401

# Import custom teleoperation configuration
try:
    from config_uarm_leader_config2 import UarmLeaderConfig2Config  # noqa: F401
    from uarm_leader_config2 import Uarm_Leader_Config2  # noqa: F401
except ImportError as e:
    logger.warning(f"Failed to import UarmLeaderConfig2 configurations: {e}")

from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (
    Robot,
    make_robot_from_config,
)
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig, LeKiwiClientConfig  # noqa: F401
from lerobot.robots.lekiwi.lekiwi import LeKiwi  # noqa: F401

from lerobot.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    gamepad,
    keyboard,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig  # noqa: F401
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig
    robot: LeKiwiConfig | LeKiwiClientConfig
    base_teleop: TeleoperatorConfig | None = None
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to display compressed images in Rerun
    display_compressed_images: bool = False


def get_base_action_from_keyboard(robot: Robot, pressed_keys: Any) -> dict[str, float]:
    """Translates pressed keyboard keys to mobile base velocity actions.
    Uses robot native helper if available, otherwise maps WASD keys.
    """
    if hasattr(robot, "_from_keyboard_to_base_action"):
        return robot._from_keyboard_to_base_action(pressed_keys)

    # Initialize state variables on robot if not present
    if not hasattr(robot, "_speed_index"):
        robot._speed_index = 1  # Start at medium (index 1 of 3)

    teleop_keys = {
        "forward": "w",
        "backward": "s",
        "left": "a",
        "right": "d",
        "rotate_left": "z",
        "rotate_right": "x",
        "speed_up": "r",
        "speed_down": "f",
    }
    speed_levels = [
        {"xy": 0.1, "theta": 30.0},
        {"xy": 0.25, "theta": 60.0},
        {"xy": 0.4, "theta": 90.0},
    ]

    # Update speed level based on keyboard events
    if teleop_keys["speed_up"] in pressed_keys:
        robot._speed_index = min(robot._speed_index + 1, len(speed_levels) - 1)
    if teleop_keys["speed_down"] in pressed_keys:
        robot._speed_index = max(robot._speed_index - 1, 0)

    speed_setting = speed_levels[robot._speed_index]
    xy_speed = speed_setting["xy"]
    theta_speed = speed_setting["theta"]

    x_cmd = 0.0
    y_cmd = 0.0
    theta_cmd = 0.0

    if teleop_keys["forward"] in pressed_keys:
        x_cmd += xy_speed
    if teleop_keys["backward"] in pressed_keys:
        x_cmd -= xy_speed
    if teleop_keys["left"] in pressed_keys:
        y_cmd += xy_speed
    if teleop_keys["right"] in pressed_keys:
        y_cmd -= xy_speed
    if teleop_keys["rotate_left"] in pressed_keys:
        theta_cmd += theta_speed
    if teleop_keys["rotate_right"] in pressed_keys:
        theta_cmd -= theta_speed

    return {
        "x.vel": x_cmd,
        "y.vel": y_cmd,
        "theta.vel": theta_cmd,
    }


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    base_teleop: Teleoperator | None = None,
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
):
    """
    This function continuously reads actions from teleoperation devices (arm leader + optional base controller),
    processes them through pipelines, sends them to the robot, and optionally displays the robot's state.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        obs = robot.get_observation()

        # Get teleop arm action
        raw_arm_action = teleop.get_action()

        # Prefix arm action keys with 'arm_' if not already present
        arm_action = {}
        for k, v in raw_arm_action.items():
            if not k.startswith("arm_") and k.endswith(".pos"):
                arm_action[f"arm_{k}"] = v
            else:
                arm_action[k] = v

        # Get base teleop action if configured
        base_action = {}
        if base_teleop is not None:
            raw_base_action = base_teleop.get_action()
            if isinstance(base_teleop, KeyboardTeleop):
                base_action = get_base_action_from_keyboard(robot, raw_base_action)
            else:
                # Map keys from generic teleoperators (e.g. gamepads) to LeKiwi base inputs
                for k, v in raw_base_action.items():
                    if k in ["x.vel", "y.vel", "theta.vel"]:
                        base_action[k] = v
                    elif k in ["linear_velocity", "linear"]:
                        base_action["x.vel"] = v
                    elif k in ["angular_velocity", "angular", "theta"]:
                        base_action["theta.vel"] = v
                    else:
                        base_action[k] = v
                
                # Ensure all required velocity fields exist
                for k in ["x.vel", "y.vel", "theta.vel"]:
                    if k not in base_action:
                        base_action[k] = 0.0
        else:
            # Maintain stationary base if no base teleoperator is active
            base_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

        # Combine arm and base actions
        raw_action = {**arm_action, **base_action}

        # Process combined action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot
        _ = robot.send_action(robot_action_to_send)

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
        move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="lekiwi_teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    
    base_teleop = None
    if cfg.base_teleop is not None:
        base_teleop = make_teleoperator_from_config(cfg.base_teleop)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()
    if base_teleop is not None:
        base_teleop.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            base_teleop=base_teleop,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        if base_teleop is not None:
            base_teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
