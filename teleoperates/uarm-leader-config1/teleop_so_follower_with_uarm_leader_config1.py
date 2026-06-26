#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import time

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig, SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

from config_uarm_leader_config1 import UarmLeaderConfig
from uarm_leader_config1 import Uarm_Leader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleoperate an SO100/SO101 follower with a uArm leader.")
    parser.add_argument("--robot", choices=("so100", "so101"), default="so101")
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="DuyKhongCay")
    parser.add_argument("--uarm-port", default="/dev/ttyUSB0")
    parser.add_argument("--uarm-id", default="DuyKhongCay")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--display-data", action="store_true")
    parser.add_argument("--no-calibrate-on-connect", action="store_true")
    return parser.parse_args()


def make_robot(args: argparse.Namespace):
    if args.robot == "so100":
        return SO100Follower(SO100FollowerConfig(port=args.robot_port, id=args.robot_id))
    return SO101Follower(SO101FollowerConfig(port=args.robot_port, id=args.robot_id))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    robot = make_robot(args)
    teleop = Uarm_Leader(
        UarmLeaderConfig(
            port=args.uarm_port,
            baudrate=args.baudrate,
            id=args.uarm_id,
            action_scale=args.scale,
        )
    )

    try:
        robot.connect()
        teleop.connect(calibrate=not args.no_calibrate_on_connect)

        log_rerun_data = None
        if args.display_data:
            from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

            init_rerun(session_name="uarm_leader_config2_teleop")

        print("Starting uArm leader teleop loop. Press Ctrl+C to stop.")
        while True:
            t0 = time.perf_counter()
            observation = robot.get_observation()
            action = teleop.get_action()
            robot.send_action(action)

            if log_rerun_data is not None:
                log_rerun_data(observation=observation, action=action)

            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\nStopping teleop.")
    finally:
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
