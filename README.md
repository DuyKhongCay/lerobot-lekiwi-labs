# LeKiwi Labs

Welcome to the **LeKiwi Labs** repository! This project serves as an experimental extension to the original [LeRobot framework](https://github.com/huggingface/lerobot), designed to introduce and test new hardware integrations, experimental features, and custom workflows.

Our goal is to expand LeRobot's capabilities by adding support for new leader arms, motors, cameras, and teleoperation interfaces tailored specifically for the LeKiwi mobile manipulator robot.

## 📂 Directory Structure
```text
lekiwi_labs/
├── cameras/         # Custom camera wrappers (IMX219 CSI, OpenCV Grayscale)
├── configs/         # Hardware configurations and default settings
├── dependencies/    # External submodules (e.g., libcamera)
├── motors/          # Motor bus drivers (e.g., Zhongli/uArm)
├── scripts/         # Executable scripts for teleoperation & testing
├── teleoperates/    # Custom teleoperators (Keyboard Omni, uArm Leader)
└── test/            # Unit tests and diagnostic tools
```

## 🚀 Experimental Features & Packages

Below is an overview of the custom packages included in this repository. Click on the respective links to read detailed documentation for each module.

### 📷 Cameras
Custom camera drivers and wrappers optimized for LeRobot, including high-performance CSI camera synchronization (IMX219) for Raspberry Pi 5 and Grayscale OpenCV processing.

### ⚙️ Motors
New motor bus implementations to bridge the communication between LeRobot and custom servo hardware.
- **Zhongli Motor Package:** Integrates Zhongli/uArm servo controller boards seamlessly with LeRobot's standardized motor bus interface (including auto-handshake and torque control).

### 🎮 Teleoperation
Experimental teleoperators for versatile control over robot arms and mobile bases.
- **uArm Leader (Config 1):** A custom leader arm integration using the uArm platform for precise robotic teleoperation (requires neutral pose calibration). *Special thanks to [MINT-SJTU/LeRobot-Anything-U-Arm](https://github.com/MINT-SJTU/LeRobot-Anything-U-Arm) for the original implementation that inspired this integration.*
  
- **Keyboard Omni:** A specialized keyboard teleoperator designed to drive an omnidirectional mobile base using gaming-style WSAD velocity commands and dynamic speed levels.

---
*Note: As an experimental extension, the packages within `lekiwi_labs` are actively being developed and tested. Please refer to each package's specific README for detailed setup, calibration, and usage instructions.*
