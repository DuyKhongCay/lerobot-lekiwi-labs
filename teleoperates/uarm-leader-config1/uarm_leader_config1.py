#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import serial

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

if __package__:
    from .config_uarm_leader_config2 import UarmLeaderConfig2Config
else:
    from config_uarm_leader_config2 import UarmLeaderConfig2Config

logger = logging.getLogger(__name__)


JOINT_SERVO_TERMS: dict[str, tuple[tuple[int, float], ...]] = {
    "shoulder_pan": ((0, -1.0),),
    "shoulder_lift": ((1, 1.0),),
    "elbow_flex": ((2, 1.0),),
    "wrist_flex": ((4, -1.0),),
    "wrist_roll": ((5, -1.0), (3, -1.0)),
    "gripper": ((6, 1.0),),
}


class UarmSerialServoReader:
    """Low-level reader for the uArm servo serial protocol used by the original scripts."""

    def __init__(self, config: UarmLeaderConfig2Config):
        if len(config.servo_ids) != 7:
            raise ValueError("Uarm_Leader_Config2 expects exactly 7 servo ids.")

        self.config = config
        self.ser: serial.Serial | None = None

    @property
    def is_connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def connect(self) -> None:
        self.ser = serial.Serial(self.config.port, self.config.baudrate, timeout=self.config.timeout_s)
        logger.info("Opened uArm leader serial port %s", self.config.port)

    @check_if_not_connected
    def configure(self) -> None:
        self.send_command("#000PVER!")
        if self.config.unlock_servos_on_connect:
            for servo_id in self.config.servo_ids:
                self.send_command("#000PCSK!")
                self.send_command(f"#{servo_id:03d}PULK!")

    @check_if_not_connected
    def send_command(self, command: str) -> str:
        assert self.ser is not None
        self.ser.write(command.encode("ascii"))
        time.sleep(self.config.command_delay_s)
        return self.ser.read_all().decode("ascii", errors="ignore")

    def pwm_to_angle(self, response: str, servo_id: int) -> float | None:
        pattern = rf"#{servo_id:03d}P(\d{{4}})"
        match = re.search(pattern, response)
        if match is None:
            return None

        pwm_value = int(match.group(1))
        pwm_span = self.config.pwm_max - self.config.pwm_min
        return (pwm_value - self.config.pwm_min) / pwm_span * self.config.angle_range_deg

    @check_if_not_connected
    def read_angles(self) -> list[float | None]:
        angles: list[float | None] = []
        for servo_id in self.config.servo_ids:
            response = self.send_command(f"#{servo_id:03d}PRAD!")
            angle = self.pwm_to_angle(response.strip(), servo_id)
            if angle is None:
                message = f"Servo {servo_id} response error: {response.strip()}"
                if self.config.strict_reads:
                    raise RuntimeError(message)
                logger.warning(message)
            angles.append(angle)
        return angles

    def disconnect(self) -> None:
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
        self.ser = None


class Uarm_Leader_Config2(Teleoperator):
    """LeRobot Teleoperator wrapper around the uArm leader serial reader."""

    config_class = UarmLeaderConfig2Config
    name = "uarm_leader_config2"

    def __init__(self, config: UarmLeaderConfig2Config):
        super().__init__(config)
        self.config = config
        self.reader = UarmSerialServoReader(config)
        self.zero_angles: list[float] | None = None
        self.logs: dict[str, float] = {}
        self.zero_angles_fpath = self.calibration_dir / f"{self.id}_zero_angles.json"

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{self.config.action_prefix}{joint}.pos": float for joint in JOINT_SERVO_TERMS}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.reader.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.zero_angles is not None and len(self.zero_angles) == len(self.config.servo_ids)

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.reader.connect()
        self.configure()

        if calibrate:
            self.calibrate()
        elif self.zero_angles_fpath.is_file():
            self._load_zero_angles()

        logger.info("%s connected.", self)

    @check_if_not_connected
    def calibrate(self) -> None:
        raw_angles = self.reader.read_angles()
        self.zero_angles = [angle if angle is not None else 0.0 for angle in raw_angles]

        if self.config.save_zero_angles:
            self._save_zero_angles()

        logger.info("uArm leader zero angles: %s", [round(angle, 2) for angle in self.zero_angles])

    @check_if_not_connected
    def configure(self) -> None:
        self.reader.configure()

    def get_action_offset(self) -> list[float]:
        if not self.is_calibrated:
            raise RuntimeError("Uarm_Leader_Config2 is not calibrated. Run `.calibrate()` first.")

        assert self.zero_angles is not None
        raw_angles = self.reader.read_angles()
        offsets: list[float] = []
        for angle, zero_angle in zip(raw_angles, self.zero_angles, strict=True):
            offsets.append(0.0 if angle is None else angle - zero_angle)
        return offsets

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        start = time.perf_counter()
        offsets = self.get_action_offset()

        action: dict[str, float] = {}
        for joint, terms in JOINT_SERVO_TERMS.items():
            value = sum(offsets[index] * sign for index, sign in terms) * self.config.action_scale
            action[f"{self.config.action_prefix}{joint}.pos"] = value

        self.logs["read_action_dt_s"] = time.perf_counter() - start
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if feedback:
            raise ValueError("Uarm_Leader_Config2 does not support force or haptic feedback.")

    @check_if_not_connected
    def disconnect(self) -> None:
        self.reader.disconnect()
        logger.info("%s disconnected.", self)

    def _load_zero_angles(self) -> None:
        with open(self.zero_angles_fpath) as f:
            zero_angles = json.load(f)

        if not isinstance(zero_angles, list) or len(zero_angles) != len(self.config.servo_ids):
            raise ValueError(f"Invalid zero-angle calibration file: {self.zero_angles_fpath}")

        self.zero_angles = [float(angle) for angle in zero_angles]

    def _save_zero_angles(self) -> None:
        assert self.zero_angles is not None
        with open(self.zero_angles_fpath, "w") as f:
            json.dump(self.zero_angles, f, indent=4)


UarmLeaderConfig2 = Uarm_Leader_Config2
