# LeKiwi Examples

sThis folder contains various standalone scripts and sample configuration files designed to help you test, configure, and understand different hardware components and networking features of the LeKiwi robot system.

## Directory Structure & Contents

### 1. Cameras (`examples/cameras/`)
Scripts to test camera functionalities and related sensors:
- **[`videos_streaming.py`](cameras/videos_streaming.py)**: A testing script for video streaming over a network using ZeroMQ (ZMQ). It demonstrates how to efficiently send and receive video frames between the robot (host) and a client machine.
- **[`imu_testing.py`](cameras/imu_testing.py)**: A utility script to test and read spatial data from the built-in IMU sensor on the IMX219-83 Stereo Camera module.

### 2. Motors (`examples/motors/`)
Scripts for actuator configuration and testing:
- **[`zhongli_set_power_on_release.py`](motors/zhongli_set_power_on_release.py)**: A setup script used to configure the "power-on-release" mode for Zhongli servos. This ensures the motors remain un-torqued (loose) when the robot boots up, which is crucial for safety and manual calibration.

### 3. Configurations (`examples/configs/`)
- This folder provides **sample configuration files (YAML)**. These files act as templates and references for running both the standard `lerobot` framework scripts and our custom LeKiwi scripts (such as teleoperation, data recording, and follower control).
