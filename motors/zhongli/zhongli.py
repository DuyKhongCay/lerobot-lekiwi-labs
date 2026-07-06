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

import logging
import re
import time
from collections.abc import Sequence

import serial

from lerobot.motors.motors_bus import (
    Motor,
    MotorCalibration,
    NameOrID,
    SerialMotorsBus,
    Value,
)

logger = logging.getLogger(__name__)


class ZhongliMotorBus(SerialMotorsBus):
    """
    Motor bus class for uArm controller, inheriting from SerialMotorsBus
    to utilize LeRobot's normalization, calibration, and recording logic.
    """

    apply_drive_mode = False
    available_baudrates = [115200]
    default_baudrate = 115200
    default_timeout = 100  # ms

    model_baudrate_table = {
        "zhongli_servo": {
            115200: 5,
        }
    }
    model_ctrl_table = {
        "zhongli_servo": {
            "Present_Position": (1, 2),
            "Goal_Position": (2, 2),
            "Homing_Offset": (3, 2),
            "Min_Position_Limit": (4, 2),
            "Max_Position_Limit": (5, 2),
            "Torque_Enable": (6, 1),
            "ID": (7, 1),
            "Baud_Rate": (8, 1),
            "Power_On_Torque_Mode": (9, 1),
        }
    }
    model_encoding_table = {}
    model_number_table = {
        "zhongli_servo": 1,
    }
    # 2667 max res translates to ~270 degrees range for 2000 PWM span (500 to 2500)
    model_resolution_table = {
        "zhongli_servo": 2667,
    }
    normalized_data = ["Goal_Position", "Present_Position"]

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ):
        super().__init__(port, motors, calibration)
        self.ser = None
        self.command_delay_s = 0.008
        self._last_goals = {}

    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        pass

    def _handshake(self, max_retries: int = 5, retry_delay_s: float = 0.3) -> None:
        # Flush any leftover bytes from the input buffer before handshaking
        self.ser.reset_input_buffer()
        for attempt in range(max_retries):
            res = self.send_command("#000PVER!")
            if res and res.strip():
                logger.info("Zhongli uArm handshake OK (attempt %d): %s", attempt + 1, res.strip())
                return
            logger.debug("Handshake attempt %d/%d: no response, retrying...", attempt + 1, max_retries)
            time.sleep(retry_delay_s)
        raise RuntimeError("Zhongli uArm board handshake failed: no response after %d attempts" % max_retries)

    def ping(self, motor: NameOrID, num_retry: int = 0, raise_on_error: bool = False) -> int | None:
        id_ = self._get_motor_id(motor)
        response = self.send_command(f"#{id_:03d}PRAD!")
        if response and response.strip().startswith(f"#{id_:03d}"):
            return 1  # model number for zhongli_servo
        if raise_on_error:
            raise ConnectionError(f"Could not ping motor {id_}")
        return None

    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        responding = {}
        for m_id in self.ids:
            if self.ping(m_id) is not None:
                responding[m_id] = 1
        return responding

    @property
    def is_connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def _connect(self, handshake: bool = True) -> None:
        self.connect(handshake=handshake)

    def connect(self, handshake: bool = True) -> None:
        try:
            self.ser = serial.Serial(self.port, self.default_baudrate, timeout=self.default_timeout / 1000.0)
            logger.info("Opened Zhongli/uArm serial port %s", self.port)
            # PL2303 and similar USB-serial adapters may reset the device on open;
            # wait for the board to finish booting before sending commands.
            time.sleep(1.5)
            if handshake:
                self._handshake()
            self.configure_motors()
        except Exception as e:
            raise ConnectionError(f"Could not connect to Zhongli/uArm on port '{self.port}'") from e

    def disconnect(self, disable_torque: bool = True) -> None:
        if disable_torque:
            self.disable_torque()
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
        self.ser = None
        logger.info("ZhongliMotorBus disconnected.")

    def get_baudrate(self) -> int:
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("ZhongliMotorBus is not connected.")
        return self.ser.baudrate

    def set_baudrate(self, baudrate: int) -> None:
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("ZhongliMotorBus is not connected.")
        if self.ser.baudrate != baudrate:
            logger.info("Setting bus baud rate to %d. Previously %d.", baudrate, self.ser.baudrate)
            self.ser.baudrate = baudrate

    def configure_motors(self) -> None:
        # Unlock all servos so they can be moved freely by default (disable torque)
        self.disable_torque()

    def disable_torque(self, motors: NameOrID | list[NameOrID] | None = None, num_retry: int = 0) -> None:
        motor_names = self._get_motors_list(motors)
        for motor in motor_names:
            m_id = self.motors[motor].id
            self.send_command("#000PCSK!")
            self.send_command(f"#{m_id:03d}PULK!")

    def _disable_torque(self, motor: int, model: str, num_retry: int = 0) -> None:
        self.send_command("#000PCSK!")
        self.send_command(f"#{motor:03d}PULK!")

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        model = self.motors[motor].model
        search_baudrates = (
            [initial_baudrate] if initial_baudrate is not None else self.available_baudrates
        )
        for baudrate in search_baudrates:
            id_model = self.broadcast_ping()
            if id_model:
                found_id = next(iter(id_model.keys()))
                return baudrate, found_id
        raise RuntimeError(f"Motor '{motor}' (model '{model}') was not found. Make sure it is connected.")

    def enable_torque(self, motors: NameOrID | list[NameOrID] | None = None, num_retry: int = 0) -> None:
        # Writing goal position automatically locks/enables torque on uArm
        motor_names = self._get_motors_list(motors)
        positions = self.sync_read("Present_Position", motor_names, normalize=False)
        for motor in motor_names:
            m_id = self.motors[motor].id
            pwm_val = int(positions[motor])
            self.send_command(f"#{m_id:03d}P{pwm_val:04d}")

    def send_command(self, command: str) -> str:
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("ZhongliMotorBus is not connected.")
        self.ser.write(command.encode("ascii"))
        time.sleep(self.command_delay_s)
        return self.ser.read_all().decode("ascii", errors="ignore")

    @property
    def is_connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    @property
    def is_calibrated(self) -> bool:
        if not self.calibration:
            return False
        return len(self.calibration) == len(self.motors)

    def read_calibration(self) -> dict[str, MotorCalibration]:
        offsets, mins, maxes = {}, {}, {}
        for motor in self.motors:
            mins[motor] = self.read("Min_Position_Limit", motor, normalize=False)
            maxes[motor] = self.read("Max_Position_Limit", motor, normalize=False)
            offsets[motor] = self.read("Homing_Offset", motor, normalize=False)

        calibration = {}
        for motor, m in self.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=int(offsets[motor]),
                range_min=int(mins[motor]),
                range_max=int(maxes[motor]),
            )
        return calibration

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        for motor, calibration in calibration_dict.items():
            self.write("Homing_Offset", motor, calibration.homing_offset)
            self.write("Min_Position_Limit", motor, calibration.range_min)
            self.write("Max_Position_Limit", motor, calibration.range_max)

        if cache:
            self.calibration = calibration_dict

    def _read(self, address: int, length: int, motor_id: int, **kwargs) -> tuple[int, int, int]:
        motor_name = self._id_to_name_dict.get(motor_id)
        
        if address == 1:  # Present_Position
            response = self.send_command(f"#{motor_id:03d}PRAD!")
            pattern = rf"#{motor_id:03d}P(\d{{4}})"
            match = re.search(pattern, response)
            if match is None:
                raise RuntimeError(f"Failed to read angle from servo {motor_id}. Response: {response}")
            raw_pwm = int(match.group(1))
            
            # Apply homing offset
            homing_offset = self.calibration[motor_name].homing_offset if (motor_name and motor_name in self.calibration) else 0
            value = raw_pwm - homing_offset
            return value, 0, 0
            
        elif address == 2:  # Goal_Position
            value = self._last_goals.get(motor_id, 1500)
            return value, 0, 0
            
        elif address == 3:  # Homing_Offset
            value = self.calibration[motor_name].homing_offset if (motor_name and motor_name in self.calibration) else 0
            return value, 0, 0
            
        elif address == 4:  # Min_Position_Limit
            value = self.calibration[motor_name].range_min if (motor_name and motor_name in self.calibration) else 0
            return value, 0, 0
            
        elif address == 5:  # Max_Position_Limit
            value = self.calibration[motor_name].range_max if (motor_name and motor_name in self.calibration) else 4095
            return value, 0, 0
            
        else:
            raise NotImplementedError(f"Unsupported read address {address}")

    def _write(self, addr: int, length: int, motor_id: int, value: int, **kwargs) -> tuple[int, int]:
        motor_name = self._id_to_name_dict.get(motor_id)
        
        if addr == 2:  # Goal_Position
            self._last_goals[motor_id] = value
            homing_offset = self.calibration[motor_name].homing_offset if (motor_name and motor_name in self.calibration) else 0
            pwm_val = int(value + homing_offset)
            pwm_val = max(500, min(2500, pwm_val))  # clamp to safe range
            self.send_command(f"#{motor_id:03d}P{pwm_val:04d}")
            return 0, 0
            
        elif addr in [3, 4, 5]:  # Calibration parameters
            return 0, 0
            
        elif addr == 6:  # Torque_Enable
            if not motor_name:
                raise ValueError(f"Cannot enable/disable torque for unregistered motor ID {motor_id}")
            if value == 1:
                self.enable_torque(motor_name)
            else:
                self.disable_torque(motor_name)
            return 0, 0
            
        elif addr == 7:  # ID
            # Send set ID command, e.g. #000PID001! to set ID to 1
            self.send_command(f"#{motor_id:03d}PID{value:03d}!")
            return 0, 0
            
        elif addr == 8:  # Baud_Rate
            # Send set baud rate command, e.g. #001PBD5!
            self.send_command(f"#{motor_id:03d}PBD{value}!")
            return 0, 0
            
        elif addr == 9:  # Power_On_Torque_Mode
            # Value 0: Power-on torque release (#PCSM!)
            # Value 1: Power-on torque restore (#PCSR!)
            if value == 0:
                self.send_command(f"#{motor_id:03d}PCSM!")
            else:
                self.send_command(f"#{motor_id:03d}PCSR!")
            time.sleep(0.1)  # Allow time for EEPROM write on the servo
            return 0, 0
            
        else:
            raise NotImplementedError(f"Unsupported write address {addr}")

    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        # Center around PWM 1500 (neutral physical position of uArm)
        half_turn_homings = {}
        for motor, pos in positions.items():
            half_turn_homings[motor] = pos - 1500
        return half_turn_homings

    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        return ids_values

    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        return ids_values

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        return [value]

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        """Find a single motor connected on the serial bus."""
        model = self.motors[motor].model
        search_baudrates = (
            [initial_baudrate] if initial_baudrate is not None else list(self.model_baudrate_table[model].keys())
        )

        for baudrate in search_baudrates:
            self.set_baudrate(baudrate)
            
            # Method 1: Try broadcast read ID (#255PID!) since only one motor is connected during setup.
            response = self.send_command("#255PID!")
            if response:
                match = re.search(r"#(\d{3})P", response)
                if match:
                    found_id = int(match.group(1))
                    if self.ping(found_id) is not None:
                        return baudrate, found_id

            # Method 2: Fallback to sequential scanning if broadcast fails
            for id_ in range(254):
                if self.ping(id_) is not None:
                    return baudrate, id_

        raise RuntimeError(f"Motor '{motor}' (model '{model}') was not found. Make sure it is connected.")

    def _disable_torque(self, motor: int, model: str, num_retry: int = 0) -> None:
        """Disable torque on a specific motor ID."""
        self.send_command(f"#{motor:03d}PULK!")

    def _sync_read(self, addr: int, length: int, motor_ids: list[int], **kwargs) -> tuple[dict[int, int], int]:
        values = {}
        for motor_id in motor_ids:
            val, _, _ = self._read(addr, length, motor_id)
            values[motor_id] = val
        return values, 0

    def _sync_write(self, addr: int, length: int, ids_values: dict[int, int], **kwargs) -> int:
        for motor_id, value in ids_values.items():
            self._write(addr, length, motor_id, value)
        return 0
