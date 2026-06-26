#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("uarm_leader")
@dataclass
class UarmLeaderConfig(TeleoperatorConfig):
    """Configuration for the uArm-as-leader serial teleoperator."""

    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout_s: float = 0.1
    command_delay_s: float = 0.008

    # The original uarm.py reads seven servos numbered 0..6.
    servo_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)

    # Original so100_teleop.py multiplies every offset by 1.5.
    action_scale: float = 1.5

    # Keep empty for SO100/SO101. Use "arm_" for LeKiwi-style action names.
    action_prefix: str = ""

    pwm_min: int = 500
    pwm_max: int = 2500
    angle_range_deg: float = 270.0
    unlock_servos_on_connect: bool = True
    strict_reads: bool = False
    save_zero_angles: bool = False
