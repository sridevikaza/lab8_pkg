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
            ret, self.camera_matrix, self.distortion_coeffs, _, _ = cv2.calibrateCamera(
                self.object_points_list, self.image_points_list, gray.shape[::-1], None, None
            )

            self.get_logger().info("Camera Matrix:\n{}".format(self.camera_matrix))
            self.get_logger().info("Distortion Coefficients:\n{}".format(self.distortion_coeffs))
            self.get_logger().info("Camera calibration completed.")

            # Call function to calculate camera mount height
            self.calculate_camera_mount_height()


    def calculate_camera_mount_height(self):
        # Pixel coordinates of the point in the image
        pixel_coordinates = np.array([665, 499])  # Reshaped to (1, 2) array
        # Distance from the camera to the object in cm
        distance_to_object = 40

        # Assuming camera matrix and distortion coefficients are available
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Assuming distortion coefficients are available as a single array
        distortion_coeffs = self.distortion_coeffs

        # Undistort the pixel coordinates
        # undistorted_pixel_coordinates = cv2.undistortPoints(pixel_coordinates,
        #                                                     self.camera_matrix, distortion_coeffs)

        # Convert pixel coordinates to normalized image coordinates
        # normalized_image_coordinates = (
        #     undistorted_pixel_coordinates.squeeze() - np.array([cx, cy])) / np.array([fx, fy])

        # Convert normalized image coordinates to homogeneous coordinates
        # homogeneous_coordinates = np.append(normalized_image_coordinates, 1)
        homogeneous_coordinates = np.append(pixel_coordinates, 1)

        # Use inverse camera matrix to convert to camera frame
        inverse_camera_matrix = np.linalg.inv(self.camera_matrix)
        point_in_camera_frame = np.dot(
            inverse_camera_matrix, homogeneous_coordinates)

        # Scale the point based on the known distance to object
        point_in_camera_frame *= distance_to_object

        # Assuming camera is at the origin, the height of the camera mount is the z-coordinate
        height_of_camera_mount = point_in_camera_frame[2]

        self.get_logger().info(
            "Height of the camera mount: {:.2f} cm".format(height_of_camera_mount))
        

def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
