#!/usr/bin/env python

# Copyright 2026 DuyKhongCay. All rights reserved.
#
# Helper script to set Power-on Torque Release for all leader servos.
# Run this once to write settings to servo EEPROM.
#

import sys
import time
from pathlib import Path

# Add project root and lerobot src to python path for importing
project_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(project_dir))

lerobot_src_dir = project_dir / "lerobot" / "src"
if lerobot_src_dir.exists():
    sys.path.append(str(lerobot_src_dir))

uarm_leader_dir = project_dir / "lekiwi_labs" / "teleoperates" / "uarm_leader_config1"
sys.path.append(str(uarm_leader_dir))

from lekiwi_labs.teleoperates.uarm_leader_config1 import UarmLeader, UarmLeaderConfig


def main():
    # Use default port
    config = UarmLeaderConfig(port="/dev/ttyUSB0")
    print(f"Connecting to uArm leader arm on port {config.port}...")
    leader_arm = UarmLeader(config)
    leader_arm.connect(calibrate=False)
    
    print("\nSetting all servos to Power-on Torque Release mode...")
    for motor_name in leader_arm.bus.motors:
        m_id = leader_arm.bus.motors[motor_name].id
        print(f"Configuring {motor_name} (ID: {m_id})...")
        leader_arm.bus.write("Power_On_Torque_Mode", motor_name, 0)
        # Sleep to let EEPROM write complete safely
        time.sleep(0.1)
    
    print("\nConfiguration complete! Disconnecting...")
    leader_arm.disconnect()
    print("Done. Now the servos will start in torque release (free) mode when powered on.")


if __name__ == "__main__":
    main()
