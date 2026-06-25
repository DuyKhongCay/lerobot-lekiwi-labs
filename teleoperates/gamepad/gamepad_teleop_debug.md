Quy trình debug thực tế nên tách theo tầng như này:

**1. Kiểm tra lỗi cấu hình trước khi cắm robot**

Hiện tại có một vấn đề lớn: [teleoperate.py](/home/duy0cay_2404/lerobot_ws/lerobot/examples/gamepad_to_so101/teleoperate.py:190) truyền:

```python
GamepadTeleopConfig(
    use_gripper=True,
    controller=args.gamepad_controller,
    use_hid=_parse_use_hid(args.use_hid),
)
```

nhưng [configuration_gamepad.py](/home/duy0cay_2404/lerobot_ws/lerobot/src/lerobot/teleoperators/gamepad/configuration_gamepad.py:24) chỉ có `use_gripper`. Tức là chạy file hiện tại có khả năng chết ngay với `unexpected keyword argument 'controller'` / `use_hid`.

Ngoài ra [teleop_gamepad.py](/home/duy0cay_2404/lerobot_ws/lerobot/src/lerobot/teleoperators/gamepad/teleop_gamepad.py:78) hiện chỉ chọn backend theo OS: macOS dùng HID, Linux dùng pygame. Nó chưa dùng `--use-hid` hay `--gamepad-controller`. Vì vậy option trong example đang chưa thật sự điều khiển backend.

**2. Debug tay cầm độc lập, chưa chạy robot**

Mục tiêu là xác nhận `gamepad.get_action()` trả đúng:

```text
delta_x, delta_y, delta_z, gripper
```

và `get_teleop_events()` trả đúng:

```text
success / failure / rerecord / intervention
```

Ở tầng này chưa gọi `robot.connect()` và chưa gọi IK. Chỉ loop đọc gamepad rồi print giá trị. Nếu joystick đảo chiều, drift, nút sai, thì sửa ở [gamepad_utils.py](/home/duy0cay_2404/lerobot_ws/lerobot/src/lerobot/teleoperators/gamepad/gamepad_utils.py:211) hoặc mapping descriptor.

**3. Xác định backend thật sự đang dùng**

Trên Linux hiện tại sẽ vào `GamepadController` pygame ở [gamepad_utils.py](/home/duy0cay_2404/lerobot_ws/lerobot/src/lerobot/teleoperators/gamepad/gamepad_utils.py:211).

HID path ở [gamepad_utils.py](/home/duy0cay_2404/lerobot_ws/lerobot/src/lerobot/teleoperators/gamepad/gamepad_utils.py:317) hiện hard-code kiểu Logitech, chưa dùng các descriptor trong [gamepad_report_descriptors.py](/home/duy0cay_2404/lerobot_ws/lerobot/src/lerobot/teleoperators/gamepad/gamepad_report_descriptors.py:188). File descriptor có `dunefox`, nhưng code đọc HID hiện tại chưa gọi `get_hid_report_descriptor()` hay `parse_hid_gamepad_report()`.

Vì vậy nếu debug DuneFox, cần sửa trước: thêm `controller` và `use_hid` vào config, rồi cho `GamepadTeleop.connect()` truyền chúng xuống controller.

**4. Sau khi input đúng, debug pipeline gamepad → IK**

Trong [teleoperate.py](/home/duy0cay_2404/lerobot_ws/lerobot/examples/gamepad_to_so101/teleoperate.py:230), luồng thực tế là:

```text
gamepad.get_action()
→ MapDeltaActionToRobotActionStep
→ EEReferenceAndDelta
→ EEBoundsAndSafety
→ GripperVelocityToJoint
→ InverseKinematicsEEToJoints
→ robot.send_action()
```

Nếu robot không chạy nhưng gamepad có giá trị, lỗi thường nằm ở một trong các điểm:

```text
URDF sai / target_frame sai
joint_names không khớp motor_names
workspace bound quá chặt
max_ee_step_m quá nhỏ
IK không tìm được nghiệm
max_relative_target chặn lệnh
```

Nên chạy chậm trước:

```bash
python DuyKhongCay_labs/SO101/gamepad/teleoperate.py \
  --robot-port=/dev/ttyACM0 \
  --robot-id=DuyKhongCay \
  --urdf-path=../SO101/so101_new_calib.urdf \
  --gamepad-controller=dunefox \
  --use-hid=true \
  --fps=10 \
  --ee-step-m=0.0005 \
  --max-ee-step-m=0.005 \
  --display-data
```

**5. Khi chạy robot thật**

Thứ tự an toàn nên là:

1. Chạy probe gamepad riêng, xác nhận axis/nút đúng.
2. Chạy robot với `ee-step-m` rất nhỏ.
3. Cầm sẵn Ctrl-C hoặc nút stop.
4. Quan sát log warning `Skipping unsafe gamepad command`.
5. Nếu có warning, debug từ `gamepad_action` sang `joint_action`, không tăng tốc vội.
6. Khi hướng di chuyển đúng mới tăng `ee-step-m`.

Tóm lại: hiện tại debug nên bắt đầu bằng việc sửa/kiểm tra tầng config-backend trước. Các option `--use-hid` và `--gamepad-controller` đang có trong example nhưng chưa được module gamepad sử dụng đúng, nên nếu chạy thực tế với DuneFox thì đây là điểm cần xử lý đầu tiên.