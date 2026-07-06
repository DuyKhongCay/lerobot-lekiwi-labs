# uArm Leader Config1

This folder contains a LeRobot `Teleoperator` implementation for the uArm leader serial controller.

Layout follows the LeRobot teleoperator pattern: `config_uarm_leader_config1.py` owns the config dataclass
and registry entry, while `uarm_leader_config1.py` owns the device implementation.

## Control Principle

The original `uarm.py` opens the uArm serial controller, unlocks servos `0..6`, reads PWM values with `#{servo_id}PRAD!`, converts PWM to degrees, and records the first pose as `zero_angles`.

Each control tick reads the current seven servo angles and computes:

```text
offset[i] = current_angle[i] - zero_angle[i]
```

The original `so100_teleop.py` maps those offsets into follower joint targets:

| Follower action | uArm source |
| --- | --- |
| `shoulder_pan.pos` | `-offset[0] * 1.5` |
| `shoulder_lift.pos` | `offset[1] * 1.5` |
| `elbow_flex.pos` | `offset[2] * 1.5` |
| `wrist_flex.pos` | `-offset[4] * 1.5` |
| `wrist_roll.pos` | `(-offset[5] - offset[3]) * 1.5` |
| `gripper.pos` | `offset[6] * 1.5` |

`Uarm_Leader_Config1` keeps that mapping but exposes it through LeRobot's `Teleoperator.get_action()` contract.

## Usage

**Important Calibration Step:** You MUST put the uArm leader in its neutral (rest) pose *before* connecting or running the script. The script uses the initial position at connection time as the zero reference. After the connection is established and calibration is complete, you can start teleoperating the robot.

To use this teleoperator in your LeRobot project, you can integrate it as a package. Ensure your environment has the necessary dependencies installed.

You can import and use the config and the teleoperator implementation as follows:

```python
from uarm_leader_config1.config_uarm_leader_config1 import UarmLeaderConfig1
from uarm_leader_config1.uarm_leader_config1 import UarmLeader

# 1. Put the uArm in its neutral pose before connecting!

# Initialize config
config = UarmLeaderConfig1(port="/dev/ttyUSB0")

# Create the teleoperator device
leader = UarmLeader(config)

# Connect and use (This sets the zero reference)
leader.connect()

# Now you can move the leader arm and get actions
action = leader.get_action()
```

### Run Example

From the workspace root, you can run the LeKiwi teleoperation script. Remember to keep the leader arm in its neutral pose when starting the script!

```bash
PYTHONPATH=lerobot/src:lekiwi_labs python3 lekiwi_labs/scripts/lekiwi_teleoperate.py \
  --robot.type lekiwi_client \
  --robot.remote_ip <ROBOT_IP_ADDRESS> \
  --teleop.port /dev/ttyUSB0
```

## Acknowledgements

Special thanks to the original implementation repository that inspired this work:
[MINT-SJTU/LeRobot-Anything-U-Arm](https://github.com/MINT-SJTU/LeRobot-Anything-U-Arm)
