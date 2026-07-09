#!/usr/bin/env python3

# Copyright 2026 LeKiwi Robot Labs. All rights reserved.
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

import atexit
from dataclasses import dataclass, field
import draccus
import logging
import signal
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from flask import Flask, Response, render_template_string
# Setup paths to ensure we can import lerobot and local lab files
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
lerobot_src = project_root / "lerobot" / "src"
if str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs

# Import grayscale camera config to ensure it registers in Draccus registry
try:
    from lekiwi_labs.cameras import grayscale_opencv
except ImportError as e:
    print(f"Warning: Could not import grayscale_opencv camera module: {e}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("lekiwi_stream")


def get_lan_ip():
    """
    Finds the active LAN IP of the current host device.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to a dummy external IP. Socket doesn't actually send packets.
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


@dataclass
class StreamConfig:
    cameras: dict[str, CameraConfig]
    host: str = "0.0.0.0"
    port: int = 5000

@draccus.wrap()
def main(cfg: StreamConfig):
    camera_configs = cfg.cameras
    for cam_name, cam_cfg in camera_configs.items():
        logger.info(f"Loaded config for camera '{cam_name}' (type: {cam_cfg.type})")

    # Initialize cameras
    logger.info("Initializing cameras...")
    cameras = make_cameras_from_configs(camera_configs)

    # Connect to cameras
    for name, camera in cameras.items():
        logger.info(f"Connecting to camera: {name} ({camera.width}x{camera.height} @ {camera.fps}fps)")
        camera.connect()
        logger.info(f"Successfully connected to camera: {name}")

    # Register cleanup on exit to release cameras cleanly
    def cleanup():
        logger.info("Starting cleanup and releasing cameras...")
        for name, camera in list(cameras.items()):
            try:
                if camera.is_connected:
                    camera.disconnect()
                    logger.info(f"Disconnected camera: {name}")
            except Exception as err:
                logger.error(f"Error disconnecting camera '{name}': {err}")

    atexit.register(cleanup)

    # Signal handlers for termination signals
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}. Exiting application...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize Flask Application
    app = Flask("lekiwi_video_stream")

    def generate_frames(cam_name):
        camera = cameras[cam_name]
        logger.info(f"Started MJPEG stream generator for camera: {cam_name}")
        
        while True:
            try:
                # Read latest processed frame (numpy array)
                frame = camera.read_latest()
                
                # Resize frame if it exceeds target streaming resolution to ensure smooth stream
                h, w = frame.shape[:2]
                target_w, target_h = 640, 480
                if w > target_w or h > target_h:
                    scale = min(target_w / w, target_h / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # LeRobot cameras default to RGB color mode. 
                # OpenCV requires BGR format to encode JPEG correctly.
                # Convert color format RGB -> BGR.
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Encode BGR image to JPEG
                ret, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                
                # Yield MJPEG chunk
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Sleep briefly based on camera FPS to prevent excessive CPU utilization
                fps = camera.fps if camera.fps else 30.0
                time.sleep(1.0 / (fps * 2.0))
                
            except Exception as ex:
                logger.debug(f"Stream frame capture failed for '{cam_name}': {ex}")
                time.sleep(0.1)

    @app.route('/video_feed/<cam_name>')
    def video_feed(cam_name):
        if cam_name not in cameras:
            return f"Camera '{cam_name}' not found.", 404
        return Response(generate_frames(cam_name), mimetype='multipart/x-mixed-replace; boundary=frame')

    # OpenCV property mapping for camera control
    OPENCV_CAMERA_PROPERTIES = {
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "saturation": cv2.CAP_PROP_SATURATION,
        "gain": cv2.CAP_PROP_GAIN,
        "exposure": cv2.CAP_PROP_EXPOSURE,
        "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE
    }

    def get_camera_properties(camera):
        """
        Gets current values for all supported OpenCV camera properties.
        """
        # OpenCVCamera has 'videocapture' attribute
        if not hasattr(camera, 'videocapture') or camera.videocapture is None:
            return {}
        
        cap = camera.videocapture
        settings: dict[str, float | None] = {}
        for name, prop_id in OPENCV_CAMERA_PROPERTIES.items():
            try:
                val = cap.get(prop_id)
                # Ensure we return clean Python types (float/int) rather than numpy types
                settings[name] = float(val) if val is not None else None
            except Exception as e:
                logger.debug(f"Could not get property {name} from {camera}: {e}")
                settings[name] = None
        return settings

    def set_camera_property(camera, name, value):
        """
        Sets a specific camera property value.
        """
        if not hasattr(camera, 'videocapture') or camera.videocapture is None:
            return False, None
            
        if name not in OPENCV_CAMERA_PROPERTIES:
            return False, None
            
        cap = camera.videocapture
        prop_id = OPENCV_CAMERA_PROPERTIES[name]
        try:
            # Map Auto Exposure value correctly
            # On Linux V4L2: 1 = Manual Mode, 3 = Aperture Priority Mode (Auto)
            if name == "auto_exposure":
                if isinstance(value, bool):
                    target_val = 3.0 if value else 1.0
                else:
                    target_val = float(value)
            else:
                target_val = float(value)
                
            success = cap.set(prop_id, target_val)
            
            # Read back actual value to verify
            actual_val = cap.get(prop_id)
            logger.info(f"Set property {name} for {camera} to {target_val}. Success: {success}, Actual: {actual_val}")
            return success, float(actual_val) if actual_val is not None else None
        except Exception as e:
            logger.error(f"Error setting property {name} to {value} on {camera}: {e}")
            return False, None

    @app.route('/api/camera/<cam_name>/controls', methods=['GET'])
    def get_controls(cam_name):
        if cam_name not in cameras:
            return {"error": f"Camera '{cam_name}' not found."}, 404
        camera = cameras[cam_name]
        settings = get_camera_properties(camera)
        return {"camera": cam_name, "settings": settings}

    @app.route('/api/camera/<cam_name>/controls', methods=['POST'])
    def set_controls(cam_name):
        from flask import request
        if cam_name not in cameras:
            return {"error": f"Camera '{cam_name}' not found."}, 404
            
        camera = cameras[cam_name]
        data = request.json or {}
        
        results = {}
        for name, value in data.items():
            if name in OPENCV_CAMERA_PROPERTIES:
                success, actual = set_camera_property(camera, name, value)
                results[name] = {"success": success, "actual": actual}
                
        return {"camera": cam_name, "results": results}

    @app.route('/')
    def index():
        # High quality UI template for streaming dashboard
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>LeKiwi Robot - Live Video Dashboard</title>
            <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
            <style>
                :root {
                    --bg-color: #0b0c10;
                    --card-bg: rgba(22, 26, 37, 0.65);
                    --accent-color: #ff5e62;
                    --accent-gradient: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
                    --text-primary: #f5f7fa;
                    --text-secondary: #8e95a5;
                    --border-color: rgba(255, 255, 255, 0.06);
                }
                
                * {
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                }
                
                body {
                    font-family: 'Outfit', sans-serif;
                    background-color: var(--bg-color);
                    color: var(--text-primary);
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                }
                
                header {
                    background: linear-gradient(180deg, rgba(0,0,0,0.4) 0%, rgba(0,0,0,0) 100%);
                    padding: 2rem;
                    text-align: center;
                    border-bottom: 1px solid var(--border-color);
                    backdrop-filter: blur(10px);
                    position: sticky;
                    top: 0;
                    z-index: 100;
                }
                
                .logo {
                    font-size: 2.2rem;
                    font-weight: 800;
                    background: var(--accent-gradient);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    margin-bottom: 0.5rem;
                    letter-spacing: 1px;
                    display: inline-block;
                    text-transform: uppercase;
                }
                
                .sub-logo {
                    font-size: 1rem;
                    color: var(--text-secondary);
                    font-weight: 300;
                }
                
                .dashboard {
                    flex: 1;
                    padding: 3rem 2rem;
                    max-width: 1400px;
                    margin: 0 auto;
                    width: 100%;
                }
                
                .grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 2.5rem;
                }
                
                @media (max-width: 600px) {
                    .grid {
                        grid-template-columns: 1fr;
                    }
                }
                
                .cam-card {
                    background: var(--card-bg);
                    border-radius: 20px;
                    border: 1px solid var(--border-color);
                    overflow: hidden;
                    backdrop-filter: blur(20px);
                    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
                    transition: transform 0.4s cubic-bezier(0.165, 0.84, 0.44, 1), box-shadow 0.4s;
                    display: flex;
                    flex-direction: column;
                }
                
                .cam-card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 20px 40px rgba(79, 172, 254, 0.15);
                    border-color: rgba(79, 172, 254, 0.3);
                }
                
                .cam-header {
                    padding: 1.2rem 1.5rem;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    background: rgba(0, 0, 0, 0.2);
                    border-bottom: 1px solid var(--border-color);
                }
                
                .cam-title {
                    font-size: 1.3rem;
                    font-weight: 600;
                    display: flex;
                    align-items: center;
                    gap: 0.6rem;
                }
                
                .status-dot {
                    width: 10px;
                    height: 10px;
                    background-color: #00f2fe;
                    border-radius: 50%;
                    display: inline-block;
                    box-shadow: 0 0 10px #00f2fe;
                    animation: blink 1.5s infinite;
                }
                
                .cam-meta {
                    font-size: 0.85rem;
                    color: var(--text-secondary);
                    background: rgba(255, 255, 255, 0.05);
                    padding: 0.3rem 0.8rem;
                    border-radius: 20px;
                    border: 1px solid var(--border-color);
                }

                .settings-toggle-btn {
                    background: rgba(255, 255, 255, 0.08);
                    border: 1px solid var(--border-color);
                    color: var(--text-primary);
                    padding: 0.4rem 0.8rem;
                    border-radius: 12px;
                    cursor: pointer;
                    font-family: inherit;
                    font-size: 0.85rem;
                    display: flex;
                    align-items: center;
                    gap: 0.4rem;
                    transition: all 0.3s ease;
                }

                .settings-toggle-btn:hover {
                    background: rgba(0, 242, 254, 0.15);
                    border-color: rgba(0, 242, 254, 0.4);
                    color: #00f2fe;
                }

                .gear-icon {
                    transition: transform 0.6s ease;
                }

                .settings-toggle-btn:hover .gear-icon {
                    transform: rotate(90deg);
                }
                
                .stream-container {
                    position: relative;
                    width: 100%;
                    padding-top: 75%; /* 4:3 Aspect Ratio */
                    background: #050608;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    overflow: hidden;
                }
                
                .stream-img {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                }

                /* Control Panel styles */
                .cam-controls {
                    background: rgba(0, 0, 0, 0.25);
                    border-top: 1px solid var(--border-color);
                    padding: 1.5rem;
                    transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
                    max-height: 500px;
                    opacity: 1;
                    overflow: hidden;
                }

                .cam-controls.collapsed {
                    max-height: 0;
                    padding-top: 0;
                    padding-bottom: 0;
                    opacity: 0;
                    border-top-color: transparent;
                    pointer-events: none;
                }

                .control-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 1.2rem;
                }

                @media (max-width: 768px) {
                    .control-grid {
                        grid-template-columns: 1fr;
                    }
                }

                .control-item {
                    display: flex;
                    flex-direction: column;
                    gap: 0.4rem;
                }

                .toggle-item {
                    flex-direction: row;
                    justify-content: space-between;
                    align-items: center;
                }

                .control-header-row {
                    display: flex;
                    justify-content: space-between;
                    font-size: 0.85rem;
                }

                .control-label {
                    color: var(--text-primary);
                    font-weight: 500;
                }

                .control-value {
                    color: #00f2fe;
                    font-weight: 600;
                    font-family: monospace;
                }

                /* Slider styling */
                input[type="range"] {
                    -webkit-appearance: none;
                    width: 100%;
                    height: 6px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 3px;
                    outline: none;
                    transition: background 0.3s;
                }

                input[type="range"]::-webkit-slider-runnable-track {
                    width: 100%;
                    height: 6px;
                    cursor: pointer;
                }

                input[type="range"]::-webkit-slider-thumb {
                    -webkit-appearance: none;
                    appearance: none;
                    width: 16px;
                    height: 16px;
                    border-radius: 50%;
                    background: #00f2fe;
                    cursor: pointer;
                    box-shadow: 0 0 10px rgba(0, 242, 254, 0.5);
                    transition: transform 0.1s, background-color 0.2s;
                    margin-top: -5px;
                }

                input[type="range"]::-webkit-slider-thumb:hover {
                    transform: scale(1.2);
                    background: #4facfe;
                }

                input[type="range"]:disabled {
                    opacity: 0.3;
                    cursor: not-allowed;
                }

                input[type="range"]:disabled::-webkit-slider-thumb {
                    background: var(--text-secondary);
                    box-shadow: none;
                    cursor: not-allowed;
                    transform: none;
                }

                /* Switch Toggle Styling */
                .switch {
                    position: relative;
                    display: inline-block;
                    width: 44px;
                    height: 24px;
                }

                .switch input {
                    opacity: 0;
                    width: 0;
                    height: 0;
                }

                .slider-round {
                    position: absolute;
                    cursor: pointer;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background-color: rgba(255, 255, 255, 0.1);
                    transition: .3s;
                    border-radius: 24px;
                    border: 1px solid var(--border-color);
                }

                .slider-round:before {
                    position: absolute;
                    content: "";
                    height: 16px;
                    width: 16px;
                    left: 3px;
                    bottom: 3px;
                    background-color: var(--text-secondary);
                    transition: .3s;
                    border-radius: 50%;
                }

                input:checked + .slider-round {
                    background: var(--accent-gradient);
                }

                input:checked + .slider-round:before {
                    transform: translateX(20px);
                    background-color: #ffffff;
                    box-shadow: 0 0 8px rgba(255, 255, 255, 0.8);
                }
                
                .cam-footer {
                    padding: 1.1rem 1.5rem;
                    background: rgba(0, 0, 0, 0.15);
                    font-size: 0.9rem;
                    color: var(--text-secondary);
                    display: flex;
                    justify-content: space-between;
                }
                
                footer {
                    padding: 2rem;
                    text-align: center;
                    font-size: 0.85rem;
                    color: var(--text-secondary);
                    border-top: 1px solid var(--border-color);
                    margin-top: auto;
                }
                
                @keyframes blink {
                    0% { opacity: 0.4; }
                    50% { opacity: 1; }
                    100% { opacity: 0.4; }
                }
            </style>
        </head>
        <body>
            <header>
                <div class="logo">LeKiwi Live Streams</div>
                <div class="sub-logo">Real-time multi-camera dashboard powered by LeRobot Framework</div>
            </header>
            
            <main class="dashboard">
                <div class="grid">
                    {% for cam_name, cam in cameras.items() %}
                    <div class="cam-card">
                        <div class="cam-header">
                            <div class="cam-title">
                                <span class="status-dot"></span>
                                Camera: {{ cam_name }}
                            </div>
                            <div style="display: flex; gap: 0.8rem; align-items: center;">
                                <span class="cam-meta">{{ camera_types[cam_name] }}</span>
                                <button class="settings-toggle-btn" onclick="toggleControls('{{ cam_name }}')">
                                    <svg class="gear-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <circle cx="12" cy="12" r="3"></circle>
                                        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                                    </svg>
                                    Settings
                                </button>
                            </div>
                        </div>
                        <div class="stream-container">
                            <img class="stream-img" src="/video_feed/{{ cam_name }}" alt="{{ cam_name }} feed">
                        </div>

                        <!-- Camera Control Panel -->
                        <div id="controls-{{ cam_name }}" class="cam-controls collapsed">
                            <div class="control-grid">
                                <!-- Auto Exposure -->
                                <div class="control-item toggle-item">
                                    <span class="control-label">Auto Exposure</span>
                                    <label class="switch">
                                        <input type="checkbox" id="auto_exposure-{{ cam_name }}" onchange="updateControl('{{ cam_name }}', 'auto_exposure', this.checked)">
                                        <span class="slider-round"></span>
                                    </label>
                                </div>
                                
                                <!-- Exposure Slider -->
                                <div class="control-item">
                                    <div class="control-header-row">
                                        <span class="control-label">Exposure</span>
                                        <span class="control-value" id="val-exposure-{{ cam_name }}">0</span>
                                    </div>
                                    <input type="range" id="exposure-{{ cam_name }}" min="1" max="10000" step="1" oninput="onSliderInput('{{ cam_name }}', 'exposure', this.value)" onchange="updateControl('{{ cam_name }}', 'exposure', this.value)">
                                </div>
                                
                                <!-- Brightness Slider -->
                                <div class="control-item">
                                    <div class="control-header-row">
                                        <span class="control-label">Brightness</span>
                                        <span class="control-value" id="val-brightness-{{ cam_name }}">0</span>
                                    </div>
                                    <input type="range" id="brightness-{{ cam_name }}" min="0" max="255" step="1" oninput="onSliderInput('{{ cam_name }}', 'brightness', this.value)" onchange="updateControl('{{ cam_name }}', 'brightness', this.value)">
                                </div>

                                <!-- Contrast Slider -->
                                <div class="control-item">
                                    <div class="control-header-row">
                                        <span class="control-label">Contrast</span>
                                        <span class="control-value" id="val-contrast-{{ cam_name }}">0</span>
                                    </div>
                                    <input type="range" id="contrast-{{ cam_name }}" min="0" max="255" step="1" oninput="onSliderInput('{{ cam_name }}', 'contrast', this.value)" onchange="updateControl('{{ cam_name }}', 'contrast', this.value)">
                                </div>

                                <!-- Saturation Slider -->
                                <div class="control-item">
                                    <div class="control-header-row">
                                        <span class="control-label">Saturation</span>
                                        <span class="control-value" id="val-saturation-{{ cam_name }}">0</span>
                                    </div>
                                    <input type="range" id="saturation-{{ cam_name }}" min="0" max="255" step="1" oninput="onSliderInput('{{ cam_name }}', 'saturation', this.value)" onchange="updateControl('{{ cam_name }}', 'saturation', this.value)">
                                </div>

                                <!-- Gain Slider -->
                                <div class="control-item">
                                    <div class="control-header-row">
                                        <span class="control-label">Gain</span>
                                        <span class="control-value" id="val-gain-{{ cam_name }}">0</span>
                                    </div>
                                    <input type="range" id="gain-{{ cam_name }}" min="0" max="255" step="1" oninput="onSliderInput('{{ cam_name }}', 'gain', this.value)" onchange="updateControl('{{ cam_name }}', 'gain', this.value)">
                                </div>
                            </div>
                        </div>

                        <div class="cam-footer">
                            <span>Resolution: {{ cam.width }}x{{ cam.height }}</span>
                            <span>Target: {{ cam.fps }} FPS</span>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </main>
            
            <footer>
                LeKiwi Labs &copy; 2026. All rights reserved.
            </footer>

            <script>
                // State to track if camera controls are loaded
                const camerasLoaded = {};
                
                // Debounce timers for each camera property API call
                const debounceTimers = {};
                
                function debounce(key, func, delay) {
                    if (debounceTimers[key]) {
                        clearTimeout(debounceTimers[key]);
                    }
                    debounceTimers[key] = setTimeout(func, delay);
                }

                function toggleControls(camName) {
                    const panel = document.getElementById(`controls-${camName}`);
                    if (!panel) return;
                    
                    const isCollapsed = panel.classList.contains('collapsed');
                    
                    if (isCollapsed) {
                        panel.classList.remove('collapsed');
                        // Load current settings from the backend API
                        loadCameraSettings(camName);
                    } else {
                        panel.classList.add('collapsed');
                    }
                }

                async function loadCameraSettings(camName) {
                    try {
                        const response = await fetch(`/api/camera/${camName}/controls`);
                        if (!response.ok) throw new Error("Failed to fetch camera controls");
                        
                        const data = await response.json();
                        const settings = data.settings;
                        
                        // Bind values to UI elements
                        for (const [key, val] of Object.entries(settings)) {
                            if (val === null || val === undefined) continue;
                            
                            if (key === 'auto_exposure') {
                                // V4L2 auto exposure: 3 is auto, 1 is manual.
                                const isAuto = (val === 3.0);
                                const checkbox = document.getElementById(`auto_exposure-${camName}`);
                                if (checkbox) {
                                    checkbox.checked = isAuto;
                                    // Disable exposure slider if auto exposure is on
                                    const expSlider = document.getElementById(`exposure-${camName}`);
                                    if (expSlider) expSlider.disabled = isAuto;
                                }
                            } else {
                                const slider = document.getElementById(`${key}-${camName}`);
                                const valDisplay = document.getElementById(`val-${key}-${camName}`);
                                if (slider) {
                                    slider.value = val;
                                }
                                if (valDisplay) {
                                    valDisplay.textContent = Math.round(val);
                                }
                            }
                        }
                        camerasLoaded[camName] = true;
                    } catch (error) {
                        console.error(`Error loading settings for camera ${camName}:`, error);
                    }
                }

                // Realtime feedback for slider movement before API is sent
                function onSliderInput(camName, propName, value) {
                    const valDisplay = document.getElementById(`val-${propName}-${camName}`);
                    if (valDisplay) {
                        valDisplay.textContent = value;
                    }
                    
                    // Debounce API update
                    debounce(`${camName}-${propName}`, () => {
                        sendControlUpdate(camName, propName, value);
                    }, 150);
                }

                async function updateControl(camName, propName, value) {
                    // This handles checkboxes (auto_exposure) which trigger change events immediately
                    if (propName === 'auto_exposure') {
                        const expSlider = document.getElementById(`exposure-${camName}`);
                        if (expSlider) {
                            expSlider.disabled = value; // value is checked state (boolean)
                        }
                        await sendControlUpdate(camName, propName, value);
                        
                        // Refresh settings after auto_exposure toggle because exposure value might be modified by driver
                        setTimeout(() => loadCameraSettings(camName), 200);
                    }
                }

                async function sendControlUpdate(camName, propName, value) {
                    try {
                        const payload = {};
                        payload[propName] = value;
                        
                        const response = await fetch(`/api/camera/${camName}/controls`, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify(payload)
                        });
                        
                        if (!response.ok) throw new Error("Failed to update control");
                        
                        const data = await response.json();
                        const result = data.results[propName];
                        
                        // Synchronize if backend returns the actual value
                        if (result && result.success && result.actual !== null) {
                            const valDisplay = document.getElementById(`val-${propName}-${camName}`);
                            const slider = document.getElementById(`${propName}-${camName}`);
                            
                            if (propName !== 'auto_exposure') {
                                if (valDisplay) valDisplay.textContent = Math.round(result.actual);
                                if (slider) slider.value = result.actual;
                            }
                        }
                    } catch (error) {
                        console.error(`Error updating control ${propName} for ${camName}:`, error);
                    }
                }
            </script>
        </body>
        </html>
        """
        camera_types = {name: cfg.type for name, cfg in camera_configs.items()}
        return render_template_string(html_template, cameras=cameras, camera_types=camera_types)

    # Launch server
    lan_ip = get_lan_ip()
    logger.info("=" * 60)
    logger.info("LeKiwi Video Stream Server is launching...")
    logger.info(f"  * Local Access:       http://localhost:{cfg.port}")
    logger.info(f"  * Network Access:     http://{lan_ip}:{cfg.port}")
    logger.info("=" * 60)

    # Disable Flask's default request logging to keep output clean and readable
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app.run(host=cfg.host, port=cfg.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

