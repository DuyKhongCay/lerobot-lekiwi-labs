import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_not_connected
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("keyboard_omni")
@dataclass
class KeyboardOmniTeleopConfig(KeyboardTeleopConfig):
    """Configuration for keyboard omni teleoperator.

    Used for controlling omnidirectional mobile robots like LeKiwi with X/Y/Theta controls.
    """

    # Keyboard keys mapping for omnidirectional base control
    teleop_keys: Dict[str, str] = field(
        default_factory=lambda: {
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
    )

    # Speed levels settings: {"xy": linear velocity in m/s, "theta": angular velocity in deg/s}
    speed_levels: List[Dict[str, float]] = field(
        default_factory=lambda: [
            {"xy": 0.1, "theta": 30.0},  # slow
            {"xy": 0.2, "theta": 60.0},  # medium
            {"xy": 0.3, "theta": 90.0},  # fast
        ]
    )

    initial_speed_index: int = 0


class KeyboardOmniTeleop(KeyboardTeleop):
    """Keyboard teleoperator for omnidirectional mobile robots like LeKiwi.

    Provides controls for driving an omnidirectional base:
    - Linear movement along X-axis (forward/backward)
    - Linear movement along Y-axis (strafe left/right)
    - Angular movement along Theta (rotation left/right)
    - Discrete speed levels adjustment
    - Disconnect/quit control

    Keyboard Controls:
        Movement:
            - W: Move forward
            - S: Move backward
            - A: Strafe left
            - D: Strafe right
            - Z: Rotate left in place
            - X: Rotate right in place

        Speed Control:
            - R: Increase speed level
            - F: Decrease speed level

        System:
            - Q / ESC: Disconnect teleoperator
    """

    config_class = KeyboardOmniTeleopConfig
    name = "keyboard_omni"

    def __init__(self, config: KeyboardOmniTeleopConfig):
        super().__init__(config)
        self.config = config
        self.teleop_keys = config.teleop_keys
        self.speed_levels = config.speed_levels
        self.speed_index = config.initial_speed_index
        # Used to detect key transitions (on-press) for speed controls and quit key
        self.prev_active_keys = set()

    @property
    def action_features(self) -> dict:
        """Return action format for omnidirectional base (linear velocities and angular velocity)."""
        return {
            "x.vel": float,
            "y.vel": float,
            "theta.vel": float,
        }

    @property
    def is_calibrated(self) -> bool:
        """Omni teleop doesn't require calibration."""
        return True

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Get the current action based on pressed keys.

        Returns:
            RobotAction with 'x.vel', 'y.vel', and 'theta.vel' keys.
        """
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        # Check which keys are currently pressed
        active_keys = {key for key, is_pressed in self.current_pressed.items() if is_pressed}

        speed_up_key = self.teleop_keys["speed_up"]
        speed_down_key = self.teleop_keys["speed_down"]
        quit_key = self.teleop_keys["quit"]

        # Adjust speed index on key transitions (press event) to avoid fast-cycling
        if speed_up_key in active_keys and speed_up_key not in self.prev_active_keys:
            self.speed_index = min(self.speed_index + 1, len(self.speed_levels) - 1)
            logging.info(
                f"Omni speed level increased to {self.speed_index}: "
                f"xy={self.speed_levels[self.speed_index]['xy']:.2f}, "
                f"theta={self.speed_levels[self.speed_index]['theta']:.2f}"
            )

        if speed_down_key in active_keys and speed_down_key not in self.prev_active_keys:
            self.speed_index = max(self.speed_index - 1, 0)
            logging.info(
                f"Omni speed level decreased to {self.speed_index}: "
                f"xy={self.speed_levels[self.speed_index]['xy']:.2f}, "
                f"theta={self.speed_levels[self.speed_index]['theta']:.2f}"
            )

        # Handle quit command
        if quit_key in active_keys and quit_key not in self.prev_active_keys:
            logging.info(f"Quit key '{quit_key}' pressed. Disconnecting.")
            self.disconnect()

        # Update previous active keys state
        self.prev_active_keys = active_keys.copy()

        # Calculate control velocities
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]
        theta_speed = speed_setting["theta"]

        x_cmd = 0.0
        y_cmd = 0.0
        theta_cmd = 0.0

        if self.teleop_keys["forward"] in active_keys:
            x_cmd += xy_speed
        if self.teleop_keys["backward"] in active_keys:
            x_cmd -= xy_speed
        if self.teleop_keys["left"] in active_keys:
            y_cmd += xy_speed
        if self.teleop_keys["right"] in active_keys:
            y_cmd -= xy_speed
        if self.teleop_keys["rotate_left"] in active_keys:
            theta_cmd += theta_speed
        if self.teleop_keys["rotate_right"] in active_keys:
            theta_cmd -= theta_speed

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
        }
