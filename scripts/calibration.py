#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
import glob


class CameraCalibrationNode(Node):
    def __init__(self):
        super().__init__('camera_calibration_node')

        self.path = '/root/sim_ws/src/'

        self.bridge = CvBridge()

        # Adjust checkerboard size if necessary
        self.checkerboard_size = (6, 8)
        self.square_width_cm = 25 / self.checkerboard_size[0]

        self.object_points = np.zeros(
            (np.prod(self.checkerboard_size), 3), dtype=np.float32)
        self.object_points[:, :2] = np.indices(
            self.checkerboard_size).T.reshape(-1, 2)
        self.object_points *= self.square_width_cm

        self.object_points_list = []
        self.image_points_list = []

        self.calibrate_camera()

    def calibrate_camera(self):
        # Path to your calibration images
        images = glob.glob(self.path + 'lab8_pkg/calibration/*.png')

        for fname in images:
            cv_image = cv2.imread(fname)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(
                gray, self.checkerboard_size, None)

            if ret:
                self.object_points_list.append(self.object_points)
                self.image_points_list.append(corners)

        if len(self.object_points_list) >= 13:  # Calibrate after collecting sufficient data
            ret, camera_matrix, distortion_coeffs, _, _ = cv2.calibrateCamera(
                self.object_points_list, self.image_points_list, gray.shape[::-1], None, None
            )

            self.get_logger().info("Camera Matrix:\n{}".format(camera_matrix))
            self.get_logger().info("Distortion Coefficients:\n{}".format(distortion_coeffs))
            self.get_logger().info("Camera calibration completed.")


def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
