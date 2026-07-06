# Zhongli Motor Package

This package provides the motor bus implementation for controlling Zhongli/uArm servo controller boards within the LeRobot framework.

## Purpose
The primary purpose of this package is to bridge the communication between LeRobot and the Zhongli/uArm hardware. By implementing the `ZhongliMotorBus` class (which inherits from LeRobot's core `SerialMotorsBus`), it allows these specific servo controllers to seamlessly integrate with LeRobot's standardized motor normalization, calibration, and teleoperation workflows.

## Features
- **Serial Communication:** Establishes robust UART serial communication with the Zhongli/uArm boards.
- **LeRobot Interface Compatibility:** Fully implements the LeRobot Motor Bus interface, allowing drop-in integration to existing robot hardware configurations.
- **Auto Handshake & Discovery:** Supports automated board handshake routines upon connection and `ping`/`broadcast_ping` commands to detect and verify active servos.
- **State Management:** Provides standardized methods to synchronously read current joint positions and write target goal positions.
- **Torque Control:** Allows programmatic enabling and disabling of motor torque during initialization and shutdown phases.
