Mình đề xuất làm theo hướng **gamepad → delta end-effector → IK → joint targets → MuJoCo `d.ctrl`**. Đây là đường sạch nhất vì repo của bạn đã có sẵn gamepad teleop và pipeline IK cho SO101 thật.

**Phương Án Chính**

1. Tạo script mới:
   [gamepad_to_so101_mujoco.py](/home/duykhongcay/lerobot_ws/DuyKhongCay_labs/mujoco/gamepad_to_so101_mujoco.py)

2. Luồng điều khiển:

```text
GamepadTeleop
  -> delta_x, delta_y, delta_z, gripper
  -> RobotKinematics dùng SO101 URDF
  -> IK ra joint action
  -> convert sang radian
  -> ghi vào MuJoCo d.ctrl
  -> mj_step + viewer.sync
```

3. Tận dụng phần đã có:
   - Gamepad: [teleop_gamepad.py](/home/duykhongcay/lerobot_ws/lerobot/src/lerobot/teleoperators/gamepad/teleop_gamepad.py)
   - Mapping DuneFox HID: [gamepad_report_descriptors.py](/home/duykhongcay/lerobot_ws/lerobot/src/lerobot/teleoperators/gamepad/gamepad_report_descriptors.py)
   - MuJoCo SO101 helper: [so101_mujoco_utils.py](/home/duykhongcay/lerobot_ws/DuyKhongCay_labs/mujoco/so101_mujoco_utils.py)
   - Ví dụ gamepad SO101 thật: [teleoperate.py](/home/duykhongcay/lerobot_ws/lerobot/examples/gamepad_to_so101/teleoperate.py)

**Mapping Tay Cầm Đề Xuất**

```text
Left stick lên/xuống     -> end-effector X
Left stick trái/phải     -> end-effector Y
Right stick lên/xuống    -> end-effector Z
LT click                 -> đóng gripper
RT click                 -> mở gripper
RB                       -> enable/intervention hold
Y                        -> success/thoát an toàn
X                        -> failure/thoát
A                        -> reset/rerecord
```

Với DuneFox, repo hiện đã có descriptor tương ứng: VID `0x04B5`, PID `0x2413`, report dài 9 bytes.

**Các Bước Triển Khai**

1. **Probe gamepad riêng**
   Chạy hoặc chỉnh từ [parse_gamepad_data.py](/home/duykhongcay/lerobot_ws/DuyKhongCay_labs/gamepad/parse_gamepad_data.py) để chắc chắn axis/nút đúng chiều.

2. **Viết adapter MuJoCo thay cho robot thật**
   Vì script hiện tại `examples/gamepad_to_so101/teleoperate.py` cần `SO101Follower` qua serial, ta nên tạo adapter mô phỏng:

```text
get_observation() -> đọc d.qpos, đổi sang degree dict
send_action()     -> nhận joint target degree, clamp limit, đổi sang rad, ghi d.ctrl
```

3. **Dùng IK Cartesian**
   Dùng `RobotKinematics` với:

```text
URDF: lerobot/SO101/so101_new_calib.urdf
target_frame: gripper_frame_link
joint_names:
  shoulder_pan
  shoulder_lift
  elbow_flex
  wrist_flex
  wrist_roll
  gripper
```

4. **Giới hạn an toàn**
   Ban đầu dùng rất chậm:

```text
fps = 30
ee_step_m = 0.0005 đến 0.001
max_ee_step_m = 0.005
workspace gần robot, ví dụ:
x: 0.05 -> 0.40
y: -0.25 -> 0.25
z: 0.02 -> 0.40
```

5. **Fallback nếu IK khó ổn định**
   Làm thêm chế độ `joint-jog` đơn giản:
   D-pad chọn joint, stick tăng/giảm góc joint. Chế độ này ít “xịn” hơn nhưng rất hữu ích để debug MuJoCo actuator, joint limit, chiều quay.

**Thứ Tự Làm Khuyến Nghị**

1. Làm script đọc gamepad và in `delta_x/y/z/gripper`.
2. Làm script gamepad → direct joint jog → `d.ctrl`.
3. Sau khi mô phỏng chạy ổn, thêm IK Cartesian.
4. Thêm safety: clamp joint range, deadzone joystick, emergency stop, reset pose.
5. Cuối cùng mới tinh chỉnh mapping để điều khiển tự nhiên hơn.

Tóm lại: nên làm bản đầu theo kiểu **MuJoCo sim adapter + GamepadTeleop có sẵn + IK pipeline của LeRobot**. Như vậy sau này bạn có thể dùng cùng logic cho cả mô phỏng và robot SO101 thật, chỉ thay backend `send_action/get_observation`.