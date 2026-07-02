Để đảm bảo độ ổn định cao nhất cho hệ thống và cài đặt thành công vào môi trường ảo (virtual environment), bạn hãy thực hiện theo quy trình chuẩn hóa sau đây:

### Bước 1: Cài đặt các gói phụ thuộc (Dependencies)

Trước khi build, hệ thống cần các công cụ biên dịch cốt lõi. Hãy mở terminal và chạy:

```bash
sudo apt update
sudo apt install -y g++ meson ninja-build pkg-config libyaml-dev python3-yaml python3-ply python3-jinja2 libgnutls28-dev openssl libudev-dev libgtest-dev
```

### Bước 2: Clone mã nguồn từ nhánh của Raspberry Pi

Vì bạn dùng phần cứng Pi, bạn bắt buộc phải dùng bản fork của Raspberry Pi thay vì bản gốc của libcamera để có đầy đủ driver (như `rpi/pisp` cho Pi 5).

```bash
git clone https://github.com/raspberrypi/libcamera.git
cd libcamera
```

### Bước 3: Cấu hình build với Meson (Tối ưu hóa)

Đây là bước quan trọng để hệ thống không phải biên dịch những module thừa. Giả sử bạn đang dùng Pi 5 (pipeline là `rpi/pisp`), hãy kích hoạt môi trường ảo (virtual environment) của bạn trước, sau đó chạy lệnh cấu hình:

```bash
# Kích hoạt môi trường ảo của bạn trước, ví dụ:
# source ~/my_env/bin/activate 
# hoặc conda activate my_env

meson setup build \
    -Dpipelines=rpi/pisp \
    -Dcam=disabled \
    -Dqcam=disabled \
    -Dtest=false \
    -Ddocumentation=disabled \
    -Dpython=enabled
```

**Kiểm tra log:** Hãy nhìn vào dòng output cấu hình Python. Nếu nó hiện **`YES`** và trỏ đúng vào đường dẫn Python trong môi trường ảo của bạn thì bạn đã cấu hình đúng.

### Bước 4: Biên dịch an toàn (Tránh OOM)

Tài liệu `pi5-camera-ubuntu` và hướng dẫn cài đặt đều cảnh báo quá trình này ngốn rất nhiều RAM. Nếu bạn chạy lệnh `ninja -C build` thông thường, nó sẽ dùng tối đa số luồng CPU và làm đứng máy ngay lập tức (đặc biệt trên các bo mạch Pi bản 4GB RAM).
Giải pháp ổn định nhất là **giới hạn số luồng biên dịch xuống 2**:

```bash
ninja -C build -j 2
```

*Lưu ý: Quá trình này sẽ tốn thời gian hơn bình thường (khoảng 10 - 15 phút), nhưng nó đảm bảo Pi của bạn không bị treo.*

### Bước 5: Tích hợp an toàn vào môi trường ảo

Sau khi quá trình biên dịch hoàn tất (chạy xong 100%), **tuyệt đối KHÔNG chạy lệnh `sudo ninja install`**. Lệnh này sẽ đẩy thư viện ra cấp độ hệ điều hành (system-wide) và có thể gây xung đột, làm mất tác dụng của môi trường ảo.

Thay vào đó, hãy lấy trực tiếp file binding vừa được tạo ra:

1. Truy cập vào thư mục chứa file Python binding vừa build:
```bash
cd build/src/py/libcamera
```

2. Tìm file có định dạng `.so` (ví dụ: `_libcamera.cpython-312-aarch64-linux-gnu.so`).
3. Copy toàn bộ thư mục `libcamera` (hoặc file `.so` này cùng các file `__init__.py`) và dán thẳng vào thư mục `site-packages` bên trong môi trường ảo của bạn.

**Ví dụ lệnh copy:**

```bash
cp -r build/src/py/libcamera /đường/dẫn/đến/môi_trường_ảo_của_bạn/lib/python3.x/site-packages/
```

Đây là quy trình an toàn, cách ly và ổn định nhất, giúp bạn có được thư viện `libcamera` (kèm Python binding) khớp 100% với phiên bản Python trong môi trường ảo, đồng thời tránh được rủi ro treo phần cứng lúc biên dịch.
