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
from dataclasses import dataclass
from typing import Any

from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
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
    action_prefix: str = ""

    pwm_min: int = 500
    pwm_max: int = 2500
    angle_range_deg: float = 270.0
    unlock_servos_on_connect: bool = True
    strict_reads: bool = False
    save_zero_angles: bool = False
    use_degrees: bool = True


# Map joint names to uArm servo indexes and their signs.
# Positive values represent scale and direction of movement relative to neutral position.
JOINT_SERVO_TERMS: dict[str, tuple[tuple[int, float], ...]] = {
    "shoulder_pan": ((0, -1.0),),
    "shoulder_lift": ((1, 1.0),),
    "elbow_flex": ((2, 1.0),),
    "wrist_flex": ((4, -1.0),),
    "wrist_roll": ((5, -1.0), (3, -1.0)),
    "gripper": ((6, 1.0),),
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
        logger.info("%s connected.", self)

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

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        start = time.perf_counter()
        
        # Read normalized positions from the bus (centered around 0)
        positions = self.bus.sync_read("Present_Position")
        
        # Map servo values to joint positions
        action: dict[str, float] = {}
        for joint, terms in JOINT_SERVO_TERMS.items():
            value = sum(positions[f"servo_{index}"] * sign for index, sign in terms) * self.config.action_scale
            action[f"{self.config.action_prefix}{joint}.pos"] = value

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
