#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team and DuyKhongCay. All rights reserved.
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
Teleoperation script for LeKiwi robot using uArm serial leader arm.
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import draccus

# Add project root and lerobot src to python path for importing
project_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(project_dir))

lerobot_src_dir = project_dir / "lerobot" / "src"
if lerobot_src_dir.exists():
    sys.path.append(str(lerobot_src_dir))

# Add uarm-leader-config1 directory to sys.path for importing uarm config and leader
uarm_leader_dir = project_dir / "lekiwi_labs" / "teleoperates" / "uarm-leader-config1"
sys.path.append(str(uarm_leader_dir))

# Import required modules
from lekiwi_labs.scripts.lekiwi_host import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from uarm_leader_config1 import UarmLeader, UarmLeaderConfig

@dataclass
class TeleoperateScriptConfig(TeleoperateConfig):
    # Use our custom LeKiwiClientConfig imported from lekiwi_host.py
    robot: LeKiwiClientConfig = field(default_factory=lambda: LeKiwiClientConfig(remote_ip="127.0.0.1"))
    # Use UarmLeaderConfig imported from uarm_leader_config1.py
    teleop: TeleoperatorConfig = field(default_factory=lambda: UarmLeaderConfig(port="/dev/ttyUSB0"))
    keyboard: KeyboardTeleopConfig = field(default_factory=KeyboardTeleopConfig)

@draccus.wrap()
def main(cfg: TeleoperateScriptConfig):
    # Initialize the robot, leader arm and keyboard teleoperators
    robot = LeKiwiClient(cfg.robot)
    leader_arm = UarmLeader(cfg.teleop)
    keyboard = KeyboardTeleop(cfg.keyboard)

    # Connect to the robot, leader arm and keyboard
    # Make sure you have the host script running on the LeKiwi robot:
    # python lekiwi_labs/scripts/lekiwi_host.py
    print(f"Connecting to LeKiwi client at {cfg.robot.remote_ip}...")
    robot.connect()
    
    print(f"Connecting to uArm leader arm on port {cfg.teleop.port}...")
    leader_arm.connect()
    
    print("Connecting to keyboard teleoperator...")
    keyboard.connect()

    # Init rerun viewer
    init_rerun(session_name="lekiwi_teleop")

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("Robot, leader arm or keyboard teleop is not connected!")

    print("Starting teleop loop...")
    try:
        while True:
            t0 = time.perf_counter()

            # Get robot observation
            observation = robot.get_observation()

            # Get teleop action from the leader arm
            arm_action = leader_arm.get_action()
            
            # Map action keys to match expected arm joint names (prefixing keys with "arm_")
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            
            # Get action from the keyboard
            keyboard_keys = keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_keys)

            # Combine arm and base actions
            action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

            # Send the combined action to the robot
            _ = robot.send_action(action)

            # Visualize the observation and action using rerun
            log_rerun_data(observation=observation, action=action)

            # Control the frequency of the teleoperation loop
            precise_sleep(max(1.0 / cfg.fps - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\nDiscontinuing teleoperation loop...")
    finally:
        print("Disconnecting devices...")
        robot.disconnect()
        leader_arm.disconnect()
        keyboard.disconnect()
        print("Teleoperation script finished cleanly.")


if __name__ == "__main__":
    main()