#!/usr/bin/env python3

# Copyright 2024 LeKiwi Labs. All rights reserved.
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
Script to test the operation of the ICM20948 IMU integrated within the IMX219StereoCamera.
This script initializes the camera with IMU support enabled, starts the hardware,
and streams IMU measurements (accelerometer, gyroscope, and fusion pose) in real-time.
"""

import sys
import time
from pathlib import Path

# Add the project workspace root to sys.path to allow importing lekiwi_labs
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lekiwi_labs.cameras.imx219_stereo_cam import IMX219StereoCamera, IMX219StereoCameraConfig


def main():
    # 1. Configure the stereo camera with IMU enabled
    config = IMX219StereoCameraConfig(
        fps=30,
        width=640,
        height=480,
        enable_imu=True,
        imu_i2c_bus=1,
    )

    print("--------------------------------------------------")
    print("IMX219 Stereo Camera & ICM20948 IMU Testing Script")
    print("--------------------------------------------------")
    print("Initializing camera and IMU connection...")

    try:
        # 2. Instantiate and connect to the camera/IMU
        camera = IMX219StereoCamera(config)
        camera.connect(warmup=True)
    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")
        print("Please check that the cameras are connected properly and python3-rtimu is installed.")
        sys.exit(1)

    print(f"Connected successfully!")
    imu_name = camera._imu.__class__.__name__
    print(f"IMU sensor name: {imu_name}")
    print("Streaming sensor data. Press Ctrl+C to exit.\n")

    last_print_time = time.time()
    frame_count = 0
    imu_count = 0

    try:
        while True:
            # Poll the IMU. Frequent polling is recommended to fetch latest data.
            imu_data = camera.read_imu()
            if imu_data is not None:
                imu_count += 1
                current_time = time.time()
                
                # Limit console update rate to ~20Hz to avoid flooding the terminal
                if current_time - last_print_time >= 0.05:
                    accel = imu_data["accel"]
                    gyro = imu_data["gyro"]
                    compass = imu_data["compass"]
                    
                    # Print formatted values in a single line using carriage return (\r)
                    print(
                        f"\r[IMU] Accel (g): [{accel[0]:6.3f}, {accel[1]:6.3f}, {accel[2]:6.3f}] | "
                        f"Gyro (°/s): [{gyro[0]:6.1f}, {gyro[1]:6.1f}, {gyro[2]:6.1f}] | "
                        f"Mag (uT): [{compass[0]:6.1f}, {compass[1]:6.1f}, {compass[2]:6.1f}]",
                        end="",
                        flush=True
                    )
                    last_print_time = current_time

            # Periodically consume camera frames from the background buffer to keep it fresh
            try:
                # read_latest returns the current frame in buffer without blocking
                _ = camera.read_latest(max_age_ms=100)
                frame_count += 1
            except (TimeoutError, RuntimeError):
                # Safe to ignore if a new frame is not ready yet or buffer is empty initially
                pass

            # Sleep briefly to reduce CPU usage while keeping a high IMU polling rate (~200Hz)
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n\nStopping test execution...")
    finally:
        # 3. Clean up and disconnect resources
        camera.disconnect()
        print(f"Stats: Captured approximately {frame_count} frames and processed {imu_count} IMU reads.")
        print("Disconnected. Test completed.")


if __name__ == "__main__":
    main()
