#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Teleoperate a SO-101 follower arm with a gamepad.

Example:

```shell
python examples/gamepad_to_so101/teleoperate.py \
    --robot-port=/dev/ttyACM0 \
    --robot-id=my_awesome_leader_arm \
    --urdf-path=./SO101/so101_new_calib.urdf \
    --gamepad-controller=dunefox
```
"""

import argparse
import logging
import time

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    MapDeltaActionToRobotActionStep,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.gamepad import GamepadTeleop, GamepadTeleopConfig
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun


def _parse_use_hid(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-port", required=True, help="Serial port of the SO-101 follower arm."
    )
    parser.add_argument(
        "--robot-id",
        default="so101_gamepad_follower",
        help="Calibration id for the robot.",
    )
    parser.add_argument(
        "--urdf-path",
        default="../so101_new_calib.urdf",
        help="Path to the SO-101 URDF used by the IK solver.",
    )
    parser.add_argument(
        "--target-frame",
        default="gripper_frame_link",
        help="URDF frame controlled by the gamepad Cartesian command.",
    )
    parser.add_argument(
        "--gamepad-controller",
        default="logitech",
        choices=("logitech", "dunefox"),
        help="Gamepad report descriptor to use.",
    )
    parser.add_argument(
        "--use-hid",
        default="auto",
        choices=("auto", "true", "false"),
        help="Use HIDAPI instead of pygame. Auto uses HID on macOS and for DuneFox.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Control loop frequency.")
    parser.add_argument(
        "--ee-step-m",
        type=float,
        default=0.002,
        help="Maximum Cartesian target offset per frame at full stick deflection.",
    )
    parser.add_argument(
        "--max-ee-step-m",
        type=float,
        default=0.03,
        help="Safety limit for one commanded end-effector position jump.",
    )
    parser.add_argument(
        "--workspace-min",
        type=float,
        nargs=3,
        default=(-1.0, -1.0, -1.0),
        metavar=("X", "Y", "Z"),
        help="Minimum allowed end-effector position.",
    )
    parser.add_argument(
        "--workspace-max",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Maximum allowed end-effector position.",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help="Per-motor safety cap passed to SO101FollowerConfig. Use <=0 to disable.",
    )
    parser.add_argument(
        "--gripper-speed-factor",
        type=float,
        default=1.0,
        help="Discrete gripper integration speed. Lower this if the gripper moves too abruptly.",
    )
    parser.add_argument(
        "--display-data", action="store_true", help="Show action data in Rerun."
    )
    return parser.parse_args()


def build_gamepad_to_joint_processor(
    robot: SO101Follower,
    urdf_path: str,
    target_frame: str,
    ee_step_m: float,
    max_ee_step_m: float,
    workspace_min: tuple[float, float, float],
    workspace_max: tuple[float, float, float],
    gripper_speed_factor: float,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    motor_names = list(robot.bus.motors.keys())
    kinematics_solver = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=target_frame,
        joint_names=motor_names,
    )

    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            MapDeltaActionToRobotActionStep(),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={
                    "x": ee_step_m,
                    "y": ee_step_m,
                    "z": ee_step_m,
                },
                motor_names=motor_names,
                use_latched_reference=False,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={
                    "min": list(workspace_min),
                    "max": list(workspace_max),
                },
                max_ee_step_m=max_ee_step_m,
            ),
            GripperVelocityToJoint(
                speed_factor=gripper_speed_factor,
                discrete_gripper=True,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=motor_names,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def main() -> None:
    args = parse_args()
    init_logging()
    logging.info("Starting SO-101 gamepad teleoperation with args: %s", vars(args))

    max_relative_target = (
        args.max_relative_target if args.max_relative_target > 0 else None
    )
    robot_config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        use_degrees=True,
        max_relative_target=max_relative_target,
    )
    gamepad_config = GamepadTeleopConfig(
        use_gripper=True,
        controller=args.gamepad_controller,
        use_hid=_parse_use_hid(args.use_hid),
    )

    robot = SO101Follower(robot_config)
    gamepad = GamepadTeleop(gamepad_config)
    gamepad_to_joint_processor = build_gamepad_to_joint_processor(
        robot=robot,
        urdf_path=args.urdf_path,
        target_frame=args.target_frame,
        ee_step_m=args.ee_step_m,
        max_ee_step_m=args.max_ee_step_m,
        workspace_min=tuple(args.workspace_min),
        workspace_max=tuple(args.workspace_max),
        gripper_speed_factor=args.gripper_speed_factor,
    )

    if args.display_data:
        init_rerun(session_name="gamepad_so101_teleop")

    try:
        robot.connect()
        gamepad.connect()
        if not robot.is_connected:
            raise RuntimeError("SO-101 follower is not connected.")
        if gamepad.gamepad is None or not gamepad.gamepad.running:
            raise RuntimeError("Gamepad is not connected or failed to start.")

        print("Starting SO-101 gamepad teleoperation. Press Ctrl-C to stop.")
        previous_gamepad_action = None
        while True:
            loop_start = time.perf_counter()

            events = gamepad.get_teleop_events()
            if events[TeleopEvents.TERMINATE_EPISODE] or events[TeleopEvents.SUCCESS]:
                print("Gamepad stop event received. Exiting teleop loop.")
                break

            robot_obs = robot.get_observation()
            robot_obs = None
            gamepad_action = gamepad.get_action()

            if gamepad_action != previous_gamepad_action:
                print("gamepad_action=", gamepad_action, flush=True)
                previous_gamepad_action = dict(gamepad_action)

            try:
                joint_action = gamepad_to_joint_processor((gamepad_action, robot_obs))
            except ValueError as exc:
                logging.warning("Skipping unsafe gamepad command: %s", exc)
                joint_action = None
            joint_action = None

            if joint_action is not None:
                print(
                    "gamepad_action=",
                    gamepad_action,
                    "| robot_obs=",
                    robot_obs,
                    "| joint_action=",
                    joint_action,
                    flush=True,
                )
                sent_action = robot.send_action(joint_action)
                if args.display_data:
                    log_rerun_data(observation=robot_obs, action=sent_action)

            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - loop_start), 0.0))

    except KeyboardInterrupt:
        pass
    finally:
        if args.display_data:
            shutdown_rerun()
        if gamepad.is_connected:
            gamepad.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
