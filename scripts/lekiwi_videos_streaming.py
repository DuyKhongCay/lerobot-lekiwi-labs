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
                            <span class="cam-meta">{{ camera_types[cam_name] }}</span>
                        </div>
                        <div class="stream-container">
                            <img class="stream-img" src="/video_feed/{{ cam_name }}" alt="{{ cam_name }} feed">
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
