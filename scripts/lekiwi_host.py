#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from functools import partialmethod
from lerobot.robots.lekiwi.lekiwi import LeKiwi
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.configs import parser

# Save original methods before they are modified by lekiwi_host.py import
original_body_to_wheel_raw = LeKiwi._body_to_wheel_raw
original_wheel_raw_to_body = LeKiwi._wheel_raw_to_body
original_configure_motors = FeetechMotorsBus.configure_motors

# Import original lekiwi_host to allow it to set up other things
import lerobot.robots.lekiwi.lekiwi_host as lekiwi_host

# Override LeKiwi kinematics parameters without modifying the original lekiwi.py script
# You can change these values to calibrate your robot's speed and rotation
WHEEL_RADIUS = 0.05  # default: 0.05 meters
BASE_RADIUS = 0.125  # default: 0.125 meters

lekiwi_host.LeKiwi._body_to_wheel_raw = partialmethod(  # type: ignore
    original_body_to_wheel_raw,
    wheel_radius=WHEEL_RADIUS,
    base_radius=BASE_RADIUS,
)
lekiwi_host.LeKiwi._wheel_raw_to_body = partialmethod(  # type: ignore
    original_wheel_raw_to_body,
    wheel_radius=WHEEL_RADIUS,
    base_radius=BASE_RADIUS,
)

# Override Feetech motors acceleration parameter (default: 254)
# You can change this value to adjust the acceleration and deceleration of the motors
ACCELERATION = 50

FeetechMotorsBus.configure_motors = partialmethod(  # type: ignore
    original_configure_motors,
    acceleration=ACCELERATION,
)

# Custom wrapper to support plugin packages discovery (such as camera types)
@parser.wrap()
def main(cfg: lekiwi_host.LeKiwiServerConfig):
    lekiwi_host.main(cfg)

if __name__ == "__main__":
    main()
