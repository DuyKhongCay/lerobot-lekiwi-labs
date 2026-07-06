#!/usr/bin/env python

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

"""
Dataset recording script for LeKiwi robot using dual teleoperation (uArm leader + Keyboard omni teleop).
"""

import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any, cast

# Patch path to import local modules
project_dir = Path(__file__).resolve().parents[2]
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

lerobot_src_dir = project_dir / "lerobot" / "src"
if lerobot_src_dir.exists() and str(lerobot_src_dir) not in sys.path:
    sys.path.append(str(lerobot_src_dir))

uarm_leader_dir = project_dir / "lekiwi_labs" / "teleoperates" / "uarm-leader-config1"
if uarm_leader_dir.exists() and str(uarm_leader_dir) not in sys.path:
    sys.path.append(str(uarm_leader_dir))

import draccus

from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import (
    make_default_processors,
)
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.scripts.lerobot_record import DatasetRecordConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# Register custom teleoperators & cameras
from lekiwi_labs.teleoperates.keyboard import KeyboardOmniTeleop, KeyboardOmniTeleopConfig  # noqa: F401
from lekiwi_labs.teleoperates.uarm_leader_config1 import UarmLeader, UarmLeaderConfig  # noqa: F401
from lekiwi_labs.cameras.grayscale_opencv import GrayscaleOpenCVCamConfig  # noqa: F401


@dataclass
class LeKiwiRecordConfig:
    robot: LeKiwiClientConfig
    dataset: DatasetRecordConfig
    teleop_arm: TeleoperatorConfig = field(
        default_factory=lambda: UarmLeaderConfig(port="/dev/ttyUSB0")
    )
    teleop_base: TeleoperatorConfig = field(
        default_factory=lambda: KeyboardOmniTeleopConfig()
    )
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False


@safe_stop_image_writer
def record_loop_lekiwi(
    robot: LeKiwiClient,
    events: dict,
    fps: int,
    teleop_arm: UarmLeader,
    teleop_keyboard: KeyboardOmniTeleop,
    teleop_action_processor: Any,
    robot_action_processor: Any,
    robot_observation_processor: Any,
    dataset: LeRobotDataset | None = None,
    control_time_s: float | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    control_interval = 1 / fps
    limit_time = float("inf") if control_time_s is None else control_time_s
    timestamp = 0
    start_episode_t = time.perf_counter()

    while timestamp < limit_time:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Applies observation processor pipeline
        obs_processed = robot_observation_processor(obs)

        # Dual teleoperation command acquisition
        arm_action = teleop_arm.get_action()
        # Map action keys to match expected arm joint names (prefixing keys with "arm_" if needed)
        arm_action = {f"arm_{k}" if not k.startswith("arm_") else k: v for k, v in arm_action.items()}

        keyboard_action = teleop_keyboard.get_action()
        base_action = keyboard_action

        # Combine arm and base actions
        act = {**arm_action, **base_action}

        act_processed_teleop = teleop_action_processor((act, obs))
        action_values = act_processed_teleop
        robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        # Send action to robot (handled over ZMQ)
        _sent_action = robot.send_action(robot_action_to_send)

        # Write to dataset
        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(
                observation=obs_processed, action=action_values, compress_images=display_compressed_images
            )

        dt_s = time.perf_counter() - start_loop_t
        sleep_time_s: float = control_interval - dt_s
        if sleep_time_s < 0:
            logging.warning(
                f"Record loop running slow ({1 / dt_s:.1f} Hz) vs target ({fps} Hz)."
            )

        precise_sleep(max(sleep_time_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t


@draccus.wrap()
def record(cfg: LeKiwiRecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    if cfg.display_data:
        init_rerun(session_name="lekiwi_recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = LeKiwiClient(cfg.robot)
    teleop_arm_cfg = cast(UarmLeaderConfig, cfg.teleop_arm)
    teleop_base_cfg = cast(KeyboardOmniTeleopConfig, cfg.teleop_base)

    teleop_arm = UarmLeader(teleop_arm_cfg)
    teleop_base = KeyboardOmniTeleop(teleop_base_cfg)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Pre-configure dataset features combining action and observation
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None

    try:
        num_cameras = len(robot.config.cameras) if (hasattr(robot, "config") and robot.config.cameras) else 0
        if cfg.resume:
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            sanity_check_dataset_name(cfg.dataset.repo_id, None)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        print(f"Connecting to LeKiwi client at {cfg.robot.remote_ip}...")
        robot.connect()
        
        print(f"Connecting to uArm leader arm on port {getattr(teleop_arm_cfg, 'port', 'unknown')}...")
        teleop_arm.connect()

        print("Connecting to keyboard base teleoperator...")
        teleop_base.connect()

        listener, events = init_keyboard_listener()

        if not cfg.dataset.streaming_encoding:
            logging.info("Streaming encoding is disabled. Consider enabling it for faster saves.")

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {recorded_episodes}", cfg.play_sounds)
                record_loop_lekiwi(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_arm=teleop_arm,
                    teleop_keyboard=teleop_base,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                )

                # Reset phase
                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    record_loop_lekiwi(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_arm=teleop_arm,
                        teleop_keyboard=teleop_base,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_compressed_images=display_compressed_images,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop_arm.is_connected:
            teleop_arm.disconnect()
        if teleop_base.is_connected:
            teleop_base.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    record()


if __name__ == "__main__":
    main()
