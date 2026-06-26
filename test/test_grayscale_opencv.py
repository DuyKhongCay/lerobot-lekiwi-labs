import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from pi5_labs.cameras.grayscale_opencv import GrayscaleCamOpenCV, GrayscaleCamOpenCVConfig
from lerobot.cameras.configs import ColorMode

class TestGrayscaleCamOpenCV(unittest.TestCase):
    def test_postprocess_grayscale_to_rgb(self):
        config = GrayscaleCamOpenCVConfig(
            index_or_path=0,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB
        )
        cam = GrayscaleCamOpenCV(config)
        
        # Test 1: 2D Grayscale input (H, W)
        img_2d = np.ones((480, 640), dtype=np.uint8) * 128
        out_rgb = cam._postprocess_image(img_2d)
        self.assertEqual(out_rgb.shape, (480, 640, 3))
        np.testing.assert_allclose(out_rgb[:, :, 0], 128)
        np.testing.assert_allclose(out_rgb[:, :, 1], 128)
        np.testing.assert_allclose(out_rgb[:, :, 2], 128)

        # Test 2: 3D Grayscale input (H, W, 1)
        img_3d_1 = np.ones((480, 640, 1), dtype=np.uint8) * 100
        out_rgb = cam._postprocess_image(img_3d_1)
        self.assertEqual(out_rgb.shape, (480, 640, 3))
        np.testing.assert_allclose(out_rgb[:, :, 0], 100)

        # Test 3: 3D BGR input (H, W, 3) where R=G=B
        img_bgr = np.ones((480, 640, 3), dtype=np.uint8) * 200
        out_rgb = cam._postprocess_image(img_bgr)
        self.assertEqual(out_rgb.shape, (480, 640, 3))
        np.testing.assert_allclose(out_rgb[:, :, 0], 200)

    def test_postprocess_grayscale_to_bgr(self):
        config = GrayscaleCamOpenCVConfig(
            index_or_path=0,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.BGR
        )
        cam = GrayscaleCamOpenCV(config)
        
        # 2D Grayscale input (H, W)
        img_2d = np.ones((480, 640), dtype=np.uint8) * 150
        out_bgr = cam._postprocess_image(img_2d)
        self.assertEqual(out_bgr.shape, (480, 640, 3))
        np.testing.assert_allclose(out_bgr[:, :, 0], 150)

    @patch('cv2.VideoCapture')
    def test_connection_and_reading(self, mock_vc_class):
        mock_vc = MagicMock()
        mock_vc.isOpened.return_value = True
        mock_vc.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_FPS: 30.0,
        }.get(prop, 0.0)
        
        # Make read return a mock frame
        mock_frame = np.ones((480, 640), dtype=np.uint8) * 128
        mock_vc.read.return_value = (True, mock_frame)
        
        mock_vc_class.return_value = mock_vc

        config = GrayscaleCamOpenCVConfig(
            index_or_path=0,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
            warmup_s=1
        )
        
        cam = GrayscaleCamOpenCV(config)
        cam.connect()
        self.assertTrue(cam.is_connected)
        
        # Read frame
        frame = cam.read()
        self.assertEqual(frame.shape, (480, 640, 3))
        np.testing.assert_allclose(frame[:, :, 0], 128)
        
        cam.disconnect()
        self.assertFalse(cam.is_connected)

    def test_make_cameras_from_configs(self):
        from pi5_labs.cameras.grayscale_opencv import make_cameras_from_configs
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.cameras.configs import CameraConfig
        
        configs: dict[str, CameraConfig] = {
            "cam1": GrayscaleCamOpenCVConfig(
                index_or_path=0,
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB
            ),
            "cam2": OpenCVCameraConfig(
                index_or_path=1,
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB
            )
        }
        
        cameras = make_cameras_from_configs(configs)
        self.assertIn("cam1", cameras)
        self.assertIn("cam2", cameras)
        self.assertIsInstance(cameras["cam1"], GrayscaleCamOpenCV)
        
        # Test unsupported type raises ValueError
        from dataclasses import dataclass
        
        @CameraConfig.register_subclass("unsupported_mock")
        @dataclass
        class MockConfig(CameraConfig):
            pass
            
        unsupported_configs: dict[str, CameraConfig] = {
            "cam3": MockConfig(fps=30, width=640, height=480)
        }
        with self.assertRaises(ValueError):
            make_cameras_from_configs(unsupported_configs)

if __name__ == '__main__':
    unittest.main()
