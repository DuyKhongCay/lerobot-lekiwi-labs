# LeKiwi Cameras Modules

## 1. Overview
This directory contains custom camera drivers and wrappers developed for the LeKiwi project, integrating seamlessly with the LeRobot framework. It provides specific implementations for handling grayscale USB cameras via OpenCV, as well as high-performance, synchronized CSI cameras (such as the IMX219 module) for the Raspberry Pi 5 using Picamera2.

## 2. Custom OpenCV Camera
These modules feature an automatic downsampling mechanism for high-resolution video streams. This approach is necessary because the camera hardware automatically crops the image sensor at lower resolutions, which reduces the Field of View (FOV). To capture the widest possible FOV, the hardware must be configured to run at a higher resolution. The frames are then downsampled in software to 640x480 to optimize performance and reduce bandwidth while maintaining compatibility with the LeRobot pipeline. This feature is implemented in both `grayscale_opencv.py` and `imx219_single_cam.py`.

## 3. Grayscale OpenCV Camera
This module provides the `GrayscaleOpenCVCam` class (registered as `grayscaleopencv`), which extends LeRobot's base `OpenCVCamera`. 

**Key Features:**
- **Monochrome Support:** Specifically designed for grayscale cameras. It automatically converts single-channel raw frames into the 3-channel RGB/BGR formats required by the LeRobot vision pipeline.
- **Hardware Control:** Automatically configures internal hardware settings upon connection, such as disabling auto-exposure and applying predefined manual exposure and gain values, ensuring consistent lighting during data collection.

## 4. IMX219 CSI Camera
These modules provide native support for the Sony IMX219 sensor connected directly to the Raspberry Pi 5's CSI ports, leveraging the advanced `Picamera2` library for low-latency, high-performance capture.

**Single Camera Features (`imx219single`):**
- Independent frame capture with highly configurable settings (resolution, framerate, RGB/BGR color mode, and image rotation).
- Implements a settable hardware warm-up period to allow the sensor's auto-exposure algorithms to stabilize before the teleoperation or recording loop begins.

**Stereo Camera Features (`imx219stereo`):**
- **Dual Camera Wrapping:** Manages two IMX219 cameras simultaneously acting as a synchronized pair using a Server-Client architecture.
- **Software Synchronization:** Accurately matches captured frame pairs by comparing their `SensorTimestamp` metadata (at nanosecond precision). A stereo image is only outputted when both frames fall within a strict, configurable synchronization threshold (e.g., 15ms).
- **Auto Concatenation:** Automatically stitches the synchronized stereo pair into a single numpy array, supporting both horizontal and vertical concatenation modes.
- **IMU Integration:** Built-in support for reading spatial data (Accelerometer, Gyroscope, Magnetometer) if the stereo camera module is equipped with an on-board ICM20948 sensor.

If you are planning to use the IMX219 CSI camera packages (`imx219_single_cam.py` or `imx219_stereo_cam.py`) on a Raspberry Pi 5 within a Python virtual environment, you must install the `libcamera` library and its Python bindings properly to avoid system crashes and conflicts. Installation guide down below.

## 5. IMX219-83 Stereo Camera IMU Configuration

If you encounter an `[Errno 2] No such file or directory: '/dev/i2c-1'` error when running scripts with IMU enabled, it means the I2C interface is not activated on your Raspberry Pi 5.

Although the physical GPIO2 (SDA) and GPIO3 (SCL) pins are correctly connected, you still need to enable I2C in the system configuration:

### Step 1: Enable I2C on Raspberry Pi 5
1. Run the system configuration tool:
   ```bash
   sudo raspi-config
   ```
2. Navigate to **Interface Options** -> **I2C**.
3. Select **Yes** when asked if you want to enable the I2C interface.
4. Select **Finish** and reboot your Raspberry Pi:
   ```bash
   sudo reboot
   ```
*(Alternatively, you can add `dtparam=i2c_arm=on` to `/boot/firmware/config.txt` and reboot).*

### Step 2: Verify Hardware Connection
1. After rebooting, check if the I2C device file exists:
   ```bash
   ls /dev/i2c*
   ```
   *If it returns `/dev/i2c-1` (or `/dev/i2c-11`), I2C is successfully enabled.*

2. Scan the I2C bus (e.g., bus `1`) to check if the IMU sensor is detected:
   ```bash
   i2cdetect -y 1
   ```
   *If the ICM20948 sensor is correctly connected and powered, you will see the address **`68`** (or **`69`**) appear on the grid.*

### Step 3: Adjust Configuration (If Bus Number is not 1)
If your GPIO 2 & 3 pins map to a different bus (e.g., `/dev/i2c-11` instead of `/dev/i2c-1`), update the `imu_i2c_bus` parameter in your camera config initialization:
```python
config = IMX219StereoCameraConfig(
    ...
    enable_imu=True,
    imu_i2c_bus=11,  # Update to the detected bus number
)
```

## 6. CSI Camera Library Virtual Environment Installation Guide
> [!WARNING]
> This guide has currently only been tested on **Raspberry Pi 5**. If you are using a different hardware platform or an older Raspberry Pi model, please proceed with caution.

To ensure maximum system stability and successfully install into a virtual environment, please follow this standardized procedure:

### Step 1: Install Dependencies

Before building, the system requires core compilation tools. Open a terminal and run:

```bash
sudo apt update
sudo apt install -y g++ meson ninja-build pkg-config libyaml-dev python3-yaml python3-ply python3-jinja2 libgnutls28-dev openssl libudev-dev libgtest-dev
```

### Step 2: Initialize the libcamera Submodule

Since you are using Raspberry Pi hardware, you must use the Raspberry Pi fork instead of the original libcamera to get the full drivers (such as `rpi/pisp` for Pi 5). This repository is already included in the project as a git submodule.

Run the following commands from the root of the workspace to sync and enter the directory:

```bash
git submodule sync lekiwi_labs/dependencies/libcamera
git submodule update --init --recursive lekiwi_labs/dependencies/libcamera
cd lekiwi_labs/dependencies/libcamera
```

### Step 3: Configure the build with Meson (Optimization)

This is a crucial step so the system doesn't compile unnecessary modules. Assuming you are using a Pi 5 (pipeline is `rpi/pisp`), activate your virtual environment first, then run the configuration command:

```bash
# Activate your virtual environment first, for example:
# source ~/my_env/bin/activate 
# or conda activate my_env

meson setup build \
    -Dpipelines=rpi/pisp \
    -Dcam=disabled \
    -Dqcam=disabled \
    -Dtest=false \
    -Ddocumentation=disabled \
    -Dpython=enabled
```

**Check the log:** Look at the Python configuration output line. If it shows **`YES`** and points correctly to the Python path in your virtual environment, then you have configured it correctly.

### Step 4: Safe Compilation (Avoid OOM)

The `pi5-camera-ubuntu` documentation and installation guides both warn that this process consumes a lot of RAM. If you run the standard `ninja -C build` command, it will use the maximum number of CPU threads and crash/freeze the machine immediately (especially on 4GB RAM Pi boards).
The most stable solution is to **limit the number of compilation threads to 2**:

```bash
ninja -C build -j 2
```

*Note: This process will take longer than usual (about 10 - 15 minutes), but it ensures your Pi will not freeze.*

### Step 5: Safe integration into the virtual environment

After the compilation is complete (reaches 100%), **absolutely DO NOT run the `sudo ninja install` command**. This command will push the library to the system-wide OS level and may cause conflicts, rendering your virtual environment ineffective.

Instead, grab the generated binding files directly:

1. Navigate to the directory containing the newly built Python binding file:
```bash
cd build/src/py/libcamera
```

2. Look for the file with a `.so` extension (e.g., `_libcamera.cpython-312-aarch64-linux-gnu.so`).
3. Copy the entire `libcamera` directory (or this `.so` file along with the `__init__.py` files) and paste it directly into the `site-packages` directory inside your virtual environment.

**Example copy command:**

```bash
cp -r build/src/py/libcamera /path/to/your/virtual_environment/lib/python3.x/site-packages/
```

This is the safest, most isolated, and most stable procedure, ensuring you get the `libcamera` library (along with Python bindings) that matches 100% with the Python version in your virtual environment, while avoiding the risk of hardware freezes during compilation.
