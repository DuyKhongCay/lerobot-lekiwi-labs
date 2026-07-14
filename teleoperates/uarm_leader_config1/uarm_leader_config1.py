# Copyright 2026 The HuggingFace Inc. team and DuyKhongCay. All rights reserved.
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

#!/usr/bin/env python

from __future__ import annotations

import logging
import time
import json
from dataclasses import dataclass
from typing import Any

from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
from lekiwi_labs.motors.zhongli import ZhongliMotorBus

logger = logging.getLogger(__name__)


@TeleoperatorConfig.register_subclass("uarm_leader")
@dataclass
class UarmLeaderConfig(TeleoperatorConfig):
    """Configuration for the uArm-as-leader serial teleoperator."""

    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout_s: float = 0.1
    command_delay_s: float = 0.008

    # The uarm leader reads seven servos numbered 0..6.
    servo_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)

    # Action multiplier scale
    action_scale: float = 1.5

    # Keep empty for SO100/SO101. Use "arm_" for LeKiwi-style action names.
    action_prefix: str = "arm_"

    pwm_min: int = 500
    pwm_max: int = 2500
    angle_range_deg: float = 270.0
    unlock_servos_on_connect: bool = True
    strict_reads: bool = False
    save_zero_angles: bool = False
    use_degrees: bool = True

    # Parameters for mapping positions based on follower calibration data
    follower_type: str = "lekiwi_client"
    follower_id: str | None = None
    follower_max_res: int = 4096
    follower_use_degrees: bool = True


# Map joint names to uArm servo indexes and their signs.
# Positive values represent scale and direction of movement relative to neutral position.
JOINT_SERVO_TERMS: dict[str, tuple[tuple[int, float], ...]] = {
    "shoulder_pan": ((0, -1.0),),
    "shoulder_lift": ((1, 1.0),),
    "elbow_flex": ((2, 1.0),),
    "wrist_flex": ((4, -1.0),),
    "wrist_roll": ((5, -1.0), (3, -1.0)),
    "gripper": ((6, -1.0),),
}


class UarmLeader(Teleoperator):
    """LeRobot Teleoperator wrapper around the ZhongliMotorBus for uArm leader."""

    config_class = UarmLeaderConfig
    name = "uarm_leader"

    def __init__(self, config: UarmLeaderConfig):
        super().__init__(config)
        self.config = config

        norm_mode = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        
        # uArm has up to 7 servos (0 to 6).
        # Define the motor dictionary expected by SerialMotorsBus.
        motors = {
            f"servo_{i}": Motor(i, "zhongli_servo", norm_mode)
            for i in config.servo_ids
        }

        self.bus = ZhongliMotorBus(
            port=config.port,
            motors = motors,
            calibration=self.calibration,
        )
        self.logs: dict[str, float] = {}

        # Resolve follower ID (falls back to leader's ID if not specified)
        self.follower_id = config.follower_id if config.follower_id is not None else self.id
        
        # Build path to follower calibration file
        self.follower_calibration_fpath = HF_LEROBOT_CALIBRATION / "robots" / config.follower_type / f"{self.follower_id}.json"
        self.follower_calibration = {}
        
        # Load calibration data if file exists
        if self.follower_calibration_fpath.is_file():
            try:
                with open(self.follower_calibration_fpath) as f:
                    self.follower_calibration = json.load(f)
                logger.info(f"Loaded follower calibration from {self.follower_calibration_fpath}")
            except Exception as e:
                logger.error(f"Failed to load follower calibration from {self.follower_calibration_fpath}: {e}")

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{self.config.action_prefix}{joint}.pos": float for joint in JOINT_SERVO_TERMS}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info("Mismatch between calibration values in the motor and the calibration file or no calibration file found")
            self.calibrate()
        self.configure()
        logger.info("%s connected.", self)

    def configure(self) -> None:
        self.bus.configure_motors()

    @check_if_not_connected
    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        # Update and write the new homing offsets to the bus before recording ranges of motion
        temp_calibration = {}
        for motor, m in self.bus.motors.items():
            temp_calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=0,
                range_max=4095,
            )
        self.bus.write_calibration(temp_calibration)

        print(
            "Move all joints sequentially through their entire ranges of motion.\n"
            "Recording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(list(self.bus.motors))

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        start = time.perf_counter()
        
        # Read raw positions (relative to homing offset, i.e., raw_pwm - homing_offset)
        # We only perform a single hardware read here.
        positions_raw = self.bus.sync_read("Present_Position", normalize=False)
        
        # Precompute normalized positions on CPU to avoid duplicate hardware reads
        # Convert positions_raw {motor_name: value} to {motor_id: value}
        raw_ids_values = {self.bus.motors[m].id: val for m, val in positions_raw.items()}
        # Decode sign if needed (ZhongliMotorBus doesn't change it, but for compatibility)
        decoded_ids_values = self.bus._decode_sign("Present_Position", raw_ids_values)
        # Normalize on CPU
        normalized_ids_values = self.bus._normalize(decoded_ids_values)
        # Map back to motor_name
        positions_norm = {self.bus._id_to_name(id_): val for id_, val in normalized_ids_values.items()}

        action: dict[str, float] = {}
        for joint, terms in JOINT_SERVO_TERMS.items():
            follower_joint_name = f"{self.config.action_prefix}{joint}"
            
            # If follower calibration data exists for this joint, use normalized mapping
            if self.follower_calibration and follower_joint_name in self.follower_calibration:
                follower_calib = self.follower_calibration[follower_joint_name]
                follower_min = follower_calib["range_min"]
                follower_max = follower_calib["range_max"]
                
                follower_ratio_list = []
                for index, sign in terms:
                    servo_name = f"servo_{index}"
                    if servo_name in self.calibration:
                        leader_calib = self.calibration[servo_name]
                        leader_min = leader_calib.range_min
                        leader_max = leader_calib.range_max
                        
                        # Read raw position of the leader servo (already fetched)
                        val = positions_raw[servo_name]
                        
                        # Compute leader's normalized ratio in [0, 1] range
                        ratio = (val - leader_min) / (leader_max - leader_min) if leader_max != leader_min else 0.5
                        ratio = max(0.0, min(1.0, ratio))
                        
                        # Adjust ratio direction based on sign
                        if sign < 0:
                            ratio = 1.0 - ratio
                        follower_ratio_list.append(ratio)
                    else:
                        # Default to mid-range if leader servo calibration is missing
                        follower_ratio_list.append(0.5)
                
                # Compute average ratio for joints driven by multiple servos (e.g., wrist_roll)
                follower_ratio = sum(follower_ratio_list) / len(follower_ratio_list) if follower_ratio_list else 0.5
                
                # Map the ratio to the follower's expected action value
                if joint == "gripper":
                    # Gripper always uses MotorNormMode.RANGE_0_100
                    value = follower_ratio * 100.0
                else:
                    if self.config.follower_use_degrees:
                        # MotorNormMode.DEGREES: map to target angle in degrees
                        follower_raw_target = follower_ratio * (follower_max - follower_min) + follower_min
                        follower_mid = (follower_min + follower_max) / 2
                        value = (follower_raw_target - follower_mid) * 360.0 / self.config.follower_max_res
                    else:
                        # MotorNormMode.RANGE_M100_100: map to range [-100, 100]
                        value = follower_ratio * 200.0 - 100.0
                        
                action[f"{follower_joint_name}.pos"] = value
            else:
                # Fallback to the original logic if no calibration mapping exists for this joint
                # It uses the precomputed CPU-normalized positions_norm
                value = sum(positions_norm[f"servo_{index}"] * sign for index, sign in terms) * self.config.action_scale
                action[f"{follower_joint_name}.pos"] = value

        self.logs["read_action_dt_s"] = time.perf_counter() - start
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if feedback:
            raise ValueError("UarmLeader does not support force or haptic feedback.")

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info("%s disconnected.", self)
