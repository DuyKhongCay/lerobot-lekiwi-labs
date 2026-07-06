# Keyboard Omni Teleoperator

This folder contains a custom LeRobot `Teleoperator` implementation designed specifically for controlling an **omnidirectional mobile base** (such as LeKiwi) via keyboard.

It extends the base `KeyboardTeleop` provided by LeRobot but overrides the key aspects to provide velocity-based control rather than joint-position control.

## Features

This custom package (`KeyboardOmniTeleop`) is tailored for mobile robots, providing an intuitive keyboard interface for driving an omnidirectional base.

### 1. Velocity-based Control
Instead of controlling static arm joints, this package controls the mobile base movement by outputting velocity commands. It returns a `RobotAction` as a dictionary: `{"x.vel": float, "y.vel": float, "theta.vel": float}`.

### 2. Gaming-style Key Mapping (WSAD)
Uses a standard gaming/rover layout tailored for an omnidirectional chassis, allowing intuitive movement in all directions:
- `W` / `S`: Move forward / backward (X-axis)
- `A` / `D`: Strafe left / right (Y-axis)
- `Z` / `X`: Rotate in place left / right (Theta)

### 3. Dynamic Speed Levels
Implements an adjustable speed level system (slow, medium, fast) to adapt to different driving scenarios.
- Each level defines specific linear (`xy` in m/s) and angular (`theta` in deg/s) speeds.
- Users can switch speeds on the fly using `R` (Speed up) and `F` (Speed down).

### 4. Calibration-Free Operation
Since the omnidirectional base operates in pure velocity mode, it does not require an absolute starting position. The system bypasses calibration entirely (`is_calibrated` always returns `True`), allowing you to start driving immediately.

## Usage

You can use `KeyboardOmniTeleop` and its configuration class `KeyboardOmniTeleopConfig` in your teleoperation scripts. It is typically run in parallel with another teleoperator (like a leader arm) so you can drive the base with one hand and operate the arm with the other.

### Example Integration

```python
from lekiwi_labs.teleoperates.keyboard import KeyboardOmniTeleop, KeyboardOmniTeleopConfig

# Initialize config (defaults to standard WSAD mapping)
config = KeyboardOmniTeleopConfig()

# Create the teleoperator device
keyboard = KeyboardOmniTeleop(config)

# Connect
keyboard.connect()

# Inside your control loop
while True:
    base_action = keyboard.get_action()
    # base_action will contain: {"x.vel": ..., "y.vel": ..., "theta.vel": ...}
    # Send this action to your mobile robot client
```

### Controls Summary
| Key | Action |
| --- | --- |
| `W` | Move forward |
| `S` | Move backward |
| `A` | Strafe left |
| `D` | Strafe right |
| `Z` | Rotate left |
| `X` | Rotate right |
| `R` | Increase speed level |
| `F` | Decrease speed level |
| `Q` or `ESC` | Quit / Disconnect |
