#!/usr/bin/env python3

# Copyright 2024 LeKiwi Labs. All rights reserved.
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

"""
Wrapper for the IMX219 Stereo Camera module connected to Raspberry Pi 5
via two CSI ports, using the Picamera2 library.

Hardware specifications:
  - Sensor: Sony IMX219 (x2)
  - Max Resolution per camera: 3280x2464
  - Baseline Length: 60mm
  - IMU: ICM20948 (Accel/Gyro/Magnetometer 16-bit)

This module provides:
  - IMX219StereoCameraConfig: Draccus-registered configuration dataclass.
  - IMX219StereoCamera: A Camera subclass that wraps two Picamera2 instances,
    performs software frame synchronisation using SensorTimestamp metadata,
    concatenates the stereo frame pair into a single numpy array, and
    optionally reads IMU data from the on-board ICM20948.
"""

import logging

import time
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation
from lerobot.utils.errors import DeviceNotConnectedError

from .imx219_single_cam import (
    _FORMAT_MAP,
    _IMX219_MODES,
    _NS_PER_MS,
    IMX219SingleCamera,
    IMX219SingleCameraConfig,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@CameraConfig.register_subclass("imx219stereo")
@dataclass
class IMX219StereoCameraConfig(IMX219SingleCameraConfig):
    """Configuration for the IMX219 Stereo Camera module on Raspberry Pi 5.

    Both cameras share the same resolution / FPS settings. One camera acts as
    the libcamera software-sync *server* and the other as the *client*.  The
    wrapper matches captured frame pairs by comparing their ``SensorTimestamp``
    metadata fields (nanosecond precision) and only outputs a stereo image when
    both timestamps are within ``sync_threshold_ms`` of each other.

    Attributes:
        fps: Target frame rate for both cameras (must be below the sensor's
            maximum for the chosen resolution so the client can "catch up").
            Defaults to 30.
        width: Width of a *single* camera's output in pixels. Defaults to 640.
            Inherited from IMX219SingleCameraConfig.
        height: Height of a *single* camera's output in pixels. Defaults to 480.
            Inherited from IMX219SingleCameraConfig.
        server_idx: Index of the camera that will act as the sync server (usually
            camera 0 on the Pi 5 CSI connector). Defaults to 0.
        client_idx: Index of the camera that will act as the sync client.
            Defaults to 1.
        color_mode: Output colour ordering (RGB or BGR). Defaults to BGR for stereo.
        rotation: Image rotation applied after capture. Defaults to NO_ROTATION.
            Inherited from IMX219SingleCameraConfig.
        warmup_s: Seconds to wait after starting both cameras before returning
            from ``connect()``. This allows auto-exposure to settle.
            Defaults to 2.
            Inherited from IMX219SingleCameraConfig.
        sync_threshold_ms: Maximum allowed timestamp difference (ms) between a
            server frame and a client frame for them to be considered a matched
            stereo pair. Defaults to 15.0.
        concat_mode: How to combine the two frames into a single array.
            - ``"horizontal"`` (default): ``np.concatenate([left, right], axis=1)``
              → output shape (H, 2*W, 3).
            - ``"vertical"``: ``np.concatenate([left, right], axis=0)``
              → output shape (2*H, W, 3).
            - ``"none"``: Not supported for the lerobot interface; the wrapper
              still returns horizontally concatenated output in read()/async_read().
        buffer_count: Number of frame buffers allocated per camera. Increase to
            reduce dropped frames under heavy load; minimum is 4 for sync.
            Defaults to 4.
            Inherited from IMX219SingleCameraConfig.
        enable_imu: Whether to enable IMU (ICM20948) reading via the icm20948 library.
            Defaults to False.
        imu_i2c_bus: I2C bus number for the ICM20948. Defaults to 1.
    """

    # Camera indices on the CSI connector
    server_idx: int = 0
    client_idx: int = 1

    # Sync & capture tuning
    sync_threshold_ms: float = 15.0
    concat_mode: str = "horizontal"  # "horizontal" | "vertical"

    # IMU support
    enable_imu: bool = False
    imu_i2c_bus: int = 1

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        super().__post_init__()

        if self.concat_mode not in ("horizontal", "vertical"):
            raise ValueError(
                f"`concat_mode` must be 'horizontal' or 'vertical', "
                f"but '{self.concat_mode}' was provided."
            )
        if self.server_idx == self.client_idx:
            raise ValueError(
                "`server_idx` and `client_idx` must be different camera indices."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Camera Implementation
# ──────────────────────────────────────────────────────────────────────────────

class IMX219StereoCamera(IMX219SingleCamera):
    """Stereo camera wrapper for two Sony IMX219 sensors on Raspberry Pi 5.

    Uses the Picamera2 library with libcamera's software synchronisation
    (SyncMode Server / Client) to capture paired frames from both CSI cameras
    as close in time as possible.

    Frame pairs are matched by comparing ``SensorTimestamp`` metadata.  When a
    matched pair is found the two frames are concatenated into a single numpy
    array and stored in an internal double-buffer protected by a threading lock.

    The class satisfies the :class:`~lerobot.cameras.camera.Camera` interface
    and can therefore be used anywhere in the LeRobot pipeline that expects a
    camera object.

    Example::

        from lekiwi_labs.cameras.picamera2.imx219_steoreo_cam import (
            IMX219StereoCamera,
            IMX219StereoCameraConfig,
        )

        config = IMX219StereoCameraConfig(fps=30, width=640, height=480)
        with IMX219StereoCamera(config) as cam:
            frame = cam.read()      # ndarray (H, 2*W, 3) — left | right
            async_frame = cam.async_read(timeout_ms=500)

    Attributes:
        config (IMX219StereoCameraConfig): Stored configuration.
    """

    def __init__(self, config: IMX219StereoCameraConfig) -> None:
        """Initialise the stereo camera wrapper.

        Does *not* open the hardware — call :meth:`connect` (or use the context
        manager) to do that.

        Args:
            config: Configuration object for this stereo camera pair.
        """
        super().__init__(config)
        self.config: IMX219StereoCameraConfig = config  # type: ignore[assignment]

        # Effective output dimensions (set during connect)
        self.fps: int | None = config.fps
        self.width: int | None = None   # will be 2*config.width after connect
        self.height: int | None = None  # will be config.height after connect

        # Picamera2 instances (created during connect)
        self._picam_server = None  # type: Any
        self._picam_client = None  # type: Any

        # Thread-synchronisation primitives
        self._frame_lock: Lock = Lock()
        self._new_frame_event: Event = Event()

        # Latest *matched* stereo frame buffer
        self._latest_frame: NDArray[Any] | None = None
        self._latest_timestamp: float | None = None   # perf_counter seconds

        # Pending single-camera frames (held until a match is found or expired)
        self._pending_server: tuple[NDArray[Any], int] | None = None  # (array, ts_ns)
        self._pending_client: tuple[NDArray[Any], int] | None = None  # (array, ts_ns)

        # IMU handle (only populated when enable_imu=True)
        self._imu = None
        self._imu_poll_interval_s: float | None = None

        # Track whether cameras are started
        self._server_started: bool = False
        self._client_started: bool = False

        # Benchmark variables
        self._bench_lock: Lock = Lock()
        self._bench_start_time: float = time.perf_counter()
        self._total_published: int = 0
        self._sync_drops: int = 0
        self._queue_drops: int = 0
        self._latencies_ms: list[float] = []
        self._periods_ms: list[float] = []
        self._last_pub_time_ms: float | None = None
        self._frame_consumed: bool = True
        self._clock_offset_ns: int = 0
        self._clock_offset_initialized: bool = False

    # ──────────────────────────────────────────────────────────────────────────
    # Camera Interface – Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """True if both cameras are open and running."""
        return self._server_started and self._client_started


    # ──────────────────────────────────────────────────────────────────────────
    # Camera Interface – connect / disconnect
    # ──────────────────────────────────────────────────────────────────────────

    def connect(self, warmup: bool = True) -> None:
        """Open both CSI cameras and start software synchronisation.

        Steps:
        1. Import Picamera2 and libcamera controls.
        2. Create a Picamera2 instance for the server camera and configure it
           with ``SyncMode=Server``.
        3. Create a Picamera2 instance for the client camera and configure it
           with ``SyncMode=Client``.
        4. Register ``post_callback`` on both instances to receive frames
           asynchronously as they are captured.
        5. Start both cameras (client first so it is ready when the server
           begins broadcasting timing packets).
        6. Optionally initialise the IMU.
        7. Perform a warm-up period to let auto-exposure settle.

        Args:
            warmup: If True, waits until at least one matched stereo pair has
                been captured before returning.

        Raises:
            ConnectionError: If the cameras cannot be opened or configured.
            RuntimeError: If warmup completes without receiving a valid frame.
        """
        if self.is_connected:
            logger.warning(f"{self} is already connected; skipping.")
            return

        try:
            from picamera2 import Picamera2  # type: ignore
            from libcamera import controls    # type: ignore
        except ImportError as e:
            raise ConnectionError(
                "Picamera2 / libcamera is not installed. "
                "Please run: sudo apt install python3-picamera2"
            ) from e

        logger.info(f"{self} opening cameras (server={self.config.server_idx}, "
                    f"client={self.config.client_idx}) ...")

        # ------------------------------------------------------------------
        # Create Picamera2 instances
        # ------------------------------------------------------------------
        try:
            self._picam_server = Picamera2(self.config.server_idx, tuning=self.config.tuning_file)
            self._picam_client = Picamera2(self.config.client_idx, tuning=self.config.tuning_file)
        except Exception as e:
            raise ConnectionError(f"{self} failed to open cameras: {e}") from e

        # ------------------------------------------------------------------
        # Build camera configurations
        # ------------------------------------------------------------------
        pixel_format = _FORMAT_MAP[self.config.color_mode]
        w, h = self.width or 640, self.height or 480
        fps = float(self.fps or 30)

        server_ctrls = {
            "FrameRate": fps,
            "SyncMode": controls.rpi.SyncModeEnum.Server,
        }
        client_ctrls = {
            "FrameRate": fps,
            "SyncMode": controls.rpi.SyncModeEnum.Client,
        }

        try:
            server_cfg = self._picam_server.create_preview_configuration(
                main={"format": pixel_format, "size": (w, h)},
                buffer_count=self.config.buffer_count,
                controls=server_ctrls,
            )
            client_cfg = self._picam_client.create_preview_configuration(
                main={"format": pixel_format, "size": (w, h)},
                buffer_count=self.config.buffer_count,
                controls=client_ctrls,
            )

            self._picam_server.configure(server_cfg)
            self._picam_client.configure(client_cfg)
        except Exception as e:
            self._release_cameras()
            raise ConnectionError(f"{self} failed to configure cameras: {e}") from e

        # ------------------------------------------------------------------
        # Register per-frame callbacks (post_callback is called by picamera2
        # AFTER each CompletedRequest is processed by the pipeline).
        # NOTE: release() must NOT be called inside the callback – the request
        # is released by Picamera2 internally after post_callback returns.
        # ------------------------------------------------------------------
        self._picam_server.post_callback = self._server_callback
        self._picam_client.post_callback = self._client_callback

        # ------------------------------------------------------------------
        # Start cameras – client first so it is waiting when server starts
        # ------------------------------------------------------------------
        try:
            self._picam_client.start()
            self._client_started = True
            self._picam_server.start()
            self._server_started = True
        except Exception as e:
            self._release_cameras()
            raise ConnectionError(f"{self} failed to start cameras: {e}") from e

        # Set effective output dimensions (for individual camera)
        self.height = h
        self.width = w

        logger.info(
            f"{self} cameras started. "
            f"Output shape: ({self.height}, {self.width}, 3). "
            f"Sync threshold: {self.config.sync_threshold_ms} ms."
        )

        # ------------------------------------------------------------------
        # Optional IMU initialisation
        # ------------------------------------------------------------------
        if self.config.enable_imu:
            try:
                self._init_imu()
            except Exception as e:
                self._release_cameras()
                raise e

        # ------------------------------------------------------------------
        # Warm-up: wait for the first valid synchronised stereo frame
        # ------------------------------------------------------------------
        if warmup and self.config.warmup_s > 0:
            logger.info(f"{self} warming up for {self.config.warmup_s}s ...")
            deadline = time.time() + self.config.warmup_s
            while time.time() < deadline:
                self._new_frame_event.wait(timeout=0.1)
                self._new_frame_event.clear()

            with self._frame_lock:
                if self._latest_frame is None:
                    raise RuntimeError(
                        f"{self} warmup ended without receiving a synchronised "
                        "stereo frame pair.  Check that both cameras are connected."
                    )
            logger.info(f"{self} warmup complete.")

    # ──────────────────────────────────────────────────────────────────────────
    # Camera Interface – read / async_read / read_latest
    # ──────────────────────────────────────────────────────────────────────────

    def read(self) -> NDArray[Any]:
        """Capture and return the next synchronised stereo frame (blocking).

        Clears the new-frame event, then waits up to 10 s for the next matched
        pair to arrive from the background callbacks.

        Returns:
            np.ndarray of shape (H, 2*W, 3) with dtype uint8.

        Raises:
            DeviceNotConnectedError: If the cameras are not connected.
            RuntimeError: If no frame arrives within the timeout.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._new_frame_event.clear()
        return self.async_read(timeout_ms=10_000)

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Return the most recent *new* synchronised stereo frame.

        Blocks until a new matched pair is available or until ``timeout_ms``
        elapses, whichever comes first.

        Args:
            timeout_ms: Maximum time to wait for a new frame (milliseconds).
                Defaults to 200 ms.

        Returns:
            np.ndarray of shape (H, 2*W, 3) with dtype uint8.

        Raises:
            DeviceNotConnectedError: If the cameras are not connected.
            TimeoutError: If no new frame arrives within ``timeout_ms``.
            RuntimeError: If an internal inconsistency is detected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        arrived = self._new_frame_event.wait(timeout=timeout_ms / 1000.0)
        if not arrived:
            raise TimeoutError(
                f"{self} timed out waiting for a synchronised stereo frame "
                f"after {timeout_ms} ms.  Check camera connections and FPS setting."
            )

        with self._frame_lock:
            frame = self._latest_frame
            self._new_frame_event.clear()

        if frame is None:
            raise RuntimeError(
                f"{self} internal error: new_frame_event was set but "
                "latest_frame is None."
            )
        
        with self._bench_lock:
            self._frame_consumed = True
            
        return frame

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recently captured stereo frame without blocking.

        Does *not* wait for a new frame – immediately returns whatever is
        currently in the buffer.  The frame may be stale.

        Args:
            max_age_ms: If the latest frame is older than this (ms) a
                :exc:`TimeoutError` is raised.  Defaults to 500 ms.

        Returns:
            np.ndarray of shape (H, 2*W, 3) with dtype uint8.

        Raises:
            DeviceNotConnectedError: If the cameras are not connected.
            RuntimeError: If no frame has been captured yet.
            TimeoutError: If the latest frame is older than ``max_age_ms``.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._frame_lock:
            frame = self._latest_frame
            timestamp = self._latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any stereo frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest stereo frame is too stale: {age_ms:.1f} ms "
                f"(max allowed: {max_age_ms} ms)."
            )
        
        with self._bench_lock:
            self._frame_consumed = True
            
        return frame

    def disconnect(self) -> None:
        """Stop both cameras and release all resources.

        Raises:
            DeviceNotConnectedError: If the cameras are already disconnected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._release_cameras()

        # Release IMU resources if active
        if self._imu is not None:
            try:
                self._imu = None
            except Exception as e:
                logger.warning(f"{self} error releasing IMU: {e}")

        logger.info(f"{self} disconnected.")

    # ──────────────────────────────────────────────────────────────────────────
    # IMU Support
    # ──────────────────────────────────────────────────────────────────────────

    def read_imu(self) -> dict[str, Any] | None:
        """Read the latest IMU sample from the ICM20948 using the icm20948 library.

        The IMU must be enabled via ``enable_imu=True`` in the configuration
        and the camera must be connected.

        Returns:
            A dictionary with keys:
            - ``"accel"``: (ax, ay, az) in g units.
            - ``"gyro"``:  (gx, gy, gz) in degrees / second.
            - ``"compass"``: (mx, my, mz) in micro-Tesla.
            - ``"timestamp"``: time.perf_counter() reading at sample time.
            - ``"fusion_pose"``: Always None (sensor fusion not supported).
            - ``"fusion_qPose"``: Always None.
            Returns ``None`` if the IMU is not enabled or not ready.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._imu is None:
            return None

        try:
            ax, ay, az, gx, gy, gz = self._imu.read_accelerometer_gyro_data()
            mx, my, mz = self._imu.read_magnetometer_data()
            return {
                "accel": (ax, ay, az),
                "gyro": (gx, gy, gz),
                "compass": (mx, my, mz),
                "timestamp": time.perf_counter(),
                "fusion_pose": None,
                "fusion_qPose": None,
            }
        except Exception as e:
            logger.warning(f"{self} IMU read error: {e}")
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _server_callback(self, request: Any) -> None:
        """Post-callback registered on the server Picamera2 instance.

        Called by picamera2 on every completed server frame.  Extracts the
        numpy array and SensorTimestamp, then attempts to match with a pending
        client frame.

        Args:
            request: A Picamera2 ``CompletedRequest`` object.  This object
                must NOT be manually released here; picamera2 will release it
                after this callback returns.
        """
        try:
            metadata = request.get_metadata()
            ts_ns: int = metadata.get("SensorTimestamp", 0)
            # make_array creates a zero-copy numpy view; copy to detach it
            # from the request buffer before picamera2 recycles the memory.
            frame: NDArray[Any] = request.make_array("main").copy()
        except Exception as e:
            logger.debug(f"{self} server callback error: {e}")
            return

        with self._frame_lock:
            self._pending_server = (frame, ts_ns)
            self._try_match_frames()

    def _client_callback(self, request: Any) -> None:
        """Post-callback registered on the client Picamera2 instance.

        Called by picamera2 on every completed client frame.

        Args:
            request: A Picamera2 ``CompletedRequest`` object.
        """
        try:
            metadata = request.get_metadata()
            ts_ns: int = metadata.get("SensorTimestamp", 0)
            frame: NDArray[Any] = request.make_array("main").copy()
        except Exception as e:
            logger.debug(f"{self} client callback error: {e}")
            return

        with self._frame_lock:
            self._pending_client = (frame, ts_ns)
            self._try_match_frames()

    def _try_match_frames(self) -> None:
        """Attempt to match the pending server and client frames.

        Must be called while ``self._frame_lock`` is held.

        Computes the absolute difference between ``SensorTimestamp`` values
        (in nanoseconds) for the current pending pair.  If the difference is
        within ``sync_threshold_ms``, the pair is considered synchronised:
        the two frames are concatenated and stored as the latest stereo frame,
        and ``new_frame_event`` is set to wake any waiting consumer.

        If the pair does not match, the older of the two pending frames is
        discarded so that it can be replaced by the next frame from its camera.
        """
        if self._pending_server is None or self._pending_client is None:
            # Not yet received a frame from both cameras – wait.
            return

        server_frame, server_ts = self._pending_server
        client_frame, client_ts = self._pending_client

        diff_ms = abs(server_ts - client_ts) / _NS_PER_MS

        if diff_ms <= self.config.sync_threshold_ms:
            # ── Matched pair found ──────────────────────────────────────────
            server_frame_rot = self._apply_rotation(server_frame)
            client_frame_rot = self._apply_rotation(client_frame)

            self._latest_frame = (server_frame_rot, client_frame_rot)
            self._latest_timestamp = time.perf_counter()

            # Consume both pending frames
            self._pending_server = None
            self._pending_client = None

            # Signal waiting consumers
            self._new_frame_event.set()

            logger.debug(f"{self} matched pair (Δts={diff_ms:.2f} ms).")

            with self._bench_lock:
                pub_time_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                
                # Dynamic clock offset detection
                if not self._clock_offset_initialized:
                    if abs(pub_time_ns - server_ts) > 10_000_000_000:
                        self._clock_offset_ns = pub_time_ns - server_ts
                    self._clock_offset_initialized = True
                
                corrected_server_ts = server_ts + self._clock_offset_ns
                latency_ms = (pub_time_ns - corrected_server_ts) / 1e6
                if latency_ms < 0:
                    latency_ms = 0.0
                
                self._latencies_ms.append(latency_ms)
                
                # Log individual frame latency to CSV if environment variable is set
                import os
                csv_path = os.environ.get("CAMERA_BENCHMARK_CSV")
                if csv_path:
                    try:
                        if not os.path.exists(csv_path):
                            with open(csv_path, "w") as f:
                                f.write("timestamp,latency_ms\n")
                        with open(csv_path, "a") as f:
                            f.write(f"{pub_time_ns / 1e9},{latency_ms:.4f}\n")
                    except Exception as csv_err:
                        logger.debug(f"Failed to log latency to CSV: {csv_err}")
                
                pub_time_ms = pub_time_ns / 1e6
                if self._last_pub_time_ms is not None:
                    period_ms = pub_time_ms - self._last_pub_time_ms
                    self._periods_ms.append(period_ms)
                self._last_pub_time_ms = pub_time_ms
                
                if not self._frame_consumed:
                    self._queue_drops += 1
                self._frame_consumed = False
                self._total_published += 1
        else:
            # ── Frames are out-of-sync – drop the older one ─────────────────
            if server_ts < client_ts:
                logger.debug(
                    f"{self} dropped stale server frame "
                    f"(Δts={diff_ms:.2f} ms > threshold={self.config.sync_threshold_ms} ms)."
                )
                self._pending_server = None
            else:
                logger.debug(
                    f"{self} dropped stale client frame "
                    f"(Δts={diff_ms:.2f} ms > threshold={self.config.sync_threshold_ms} ms)."
                )
                self._pending_client = None
                
            with self._bench_lock:
                self._sync_drops += 1

    def get_benchmark_report(self) -> dict[str, Any]:
        """Compute camera benchmark report and reset metrics."""
        with self._bench_lock:
            now = time.perf_counter()
            duration = now - self._bench_start_time
            self._bench_start_time = now

            published = self._total_published
            sync_drops = self._sync_drops
            queue_drops = self._queue_drops
            total_dropped = sync_drops + queue_drops

            # Actual FPS
            actual_fps = published / duration if duration > 0 else 0.0

            # Drop rates
            total_frames = published + total_dropped
            drop_rate = (total_dropped / total_frames * 100) if total_frames > 0 else 0.0
            sync_drop_rate = (sync_drops / total_frames * 100) if total_frames > 0 else 0.0
            queue_drop_rate = (queue_drops / total_frames * 100) if total_frames > 0 else 0.0

            # Latency stats
            if self._latencies_ms:
                avg_latency = sum(self._latencies_ms) / len(self._latencies_ms)
                max_latency = max(self._latencies_ms)
            else:
                avg_latency = 0.0
                max_latency = 0.0

            # Jitter stats
            if self._periods_ms:
                jitter = float(np.std(self._periods_ms))
            else:
                jitter = 0.0

            # Reset metrics
            self._total_published = 0
            self._sync_drops = 0
            self._queue_drops = 0
            self._latencies_ms.clear()
            self._periods_ms.clear()

            return {
                "duration": duration,
                "target_fps": float(self.fps) if self.fps else 30.0,
                "actual_fps": actual_fps,
                "published": published,
                "total_dropped": total_dropped,
                "sync_drops": sync_drops,
                "queue_drops": queue_drops,
                "drop_rate": drop_rate,
                "sync_drop_rate": sync_drop_rate,
                "queue_drop_rate": queue_drop_rate,
                "avg_latency": avg_latency,
                "max_latency": max_latency,
                "jitter": jitter,
            }

    def _concat_frames(
        self, server_frame: NDArray[Any], client_frame: NDArray[Any]
    ) -> NDArray[Any]:
        """Concatenate two camera frames into a single stereo image array.

        Applies the rotation specified in the configuration before concatenating.

        Args:
            server_frame: Frame from the server (left) camera.
            client_frame: Frame from the client (right) camera.

        Returns:
            Concatenated numpy array.
        """
        server_frame = self._apply_rotation(server_frame)
        client_frame = self._apply_rotation(client_frame)

        if self.config.concat_mode == "horizontal":
            return np.concatenate([server_frame, client_frame], axis=1)
        else:  # "vertical"
            return np.concatenate([server_frame, client_frame], axis=0)


    def _release_cameras(self) -> None:
        """Stop and close both Picamera2 instances, clearing all state."""
        for cam, name, started_attr in [
            (self._picam_server, "server", "_server_started"),
            (self._picam_client, "client", "_client_started"),
        ]:
            if cam is not None:
                try:
                    cam.post_callback = None
                    if getattr(self, started_attr, False):
                        cam.stop()
                    cam.close()
                except Exception as e:
                    logger.warning(f"{self} error releasing {name} camera: {e}")
                finally:
                    setattr(self, started_attr, False)

        self._picam_server = None
        self._picam_client = None

        with self._frame_lock:
            self._latest_frame = None
            self._latest_timestamp = None
            self._pending_server = None
            self._pending_client = None
            self._new_frame_event.clear()

    def _init_imu(self) -> None:
        """Initialise the ICM20948 IMU via the icm20948 Python library.

        Raises:
            ImportError: If the icm20948 or smbus2 library is not installed.
            RuntimeError: If the IMU fails to initialise on the I2C bus.
        """
        try:
            from icm20948 import ICM20948  # type: ignore
            import smbus2
        except ImportError as e:
            raise ImportError(
                f"The 'icm20948' and 'smbus2' libraries are required when `enable_imu` is True. "
                "Please install them using: pip install -e .[imu]"
            ) from e

        try:
            # Try to initialize with the specified I2C bus and address.
            # Default Pimoroni address is 0x68 (sometimes 0x69).
            # The icm20948 library requires an SMBus object instance, not an integer bus ID.
            bus = smbus2.SMBus(self.config.imu_i2c_bus)
            try:
                imu = ICM20948(i2c_addr=0x68, i2c_bus=bus)
                _ = imu.read_accelerometer_gyro_data()  # Dummy read to verify
            except Exception:
                logger.info("Failed to init ICM20948 at 0x68, trying 0x69...")
                bus = smbus2.SMBus(self.config.imu_i2c_bus)
                imu = ICM20948(i2c_addr=0x69, i2c_bus=bus)
                _ = imu.read_accelerometer_gyro_data()

            self._imu = imu
            # Set a nominal poll interval of 5ms (200Hz) as icm20948 doesn't expose it
            self._imu_poll_interval_s = 0.005

            logger.info(
                f"{self} IMU initialised: ICM20948 on I2C bus {self.config.imu_i2c_bus}."
            )

        except Exception as e:
            raise RuntimeError(f"Unexpected error during IMU init: {e}") from e

    # ──────────────────────────────────────────────────────────────────────────
    # String representation
    # ──────────────────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return (
            f"IMX219StereoCamera("
            f"server={self.config.server_idx}, "
            f"client={self.config.client_idx}, "
            f"{self.config.width}x{self.config.height}@{self.config.fps}fps)"
        )
