# uArm Leader Config2

This folder contains a LeRobot `Teleoperator` implementation for the uArm leader serial controller.

Layout follows the LeRobot teleoperator pattern: `config_uarm_leader_config2.py` owns the config dataclass
and registry entry, while `uarm_leader_config2.py` owns the device implementation.

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

`Uarm_Leader_Config2` keeps that mapping but exposes it through LeRobot's `Teleoperator.get_action()` contract.

## Run

From the workspace root:

```bash
PYTHONPATH=lerobot/src python3 DuyKhongCay_labs/SO101/uarm-leader-config2/teleop_so_follower_with_uarm_leader_config2.py \
  --robot so101 \
  --robot-port /dev/ttyACM1 \
  --uarm-port /dev/ttyUSB0
```

Put the uArm leader in the neutral pose before connecting; that pose becomes the zero reference.
