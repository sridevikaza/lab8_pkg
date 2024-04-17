#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import glob

class DistanceNode(Node):
    def __init__(self):
        super().__init__('distance_node')
        self.path = '/root/sim_ws/src/'
        # load camera calibration matrix
        self.calibration_matrix, self.dist_coeffs = self.calibrate_camera(self.path+'lab8_pkg/calibration', (6, 8), 25)
        print("Calibration matrix:\n", self.calibration_matrix)
        print("Distortion coefficients:\n", self.dist_coeffs)


    def calibrate_camera(self, calibration_folder, chessboard_size, square_size):
        # prepare object points
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

        # arrays to store object points and image points from all the images
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane

        # get the list of calibration images
        calibration_images = glob.glob(calibration_folder + '/*.png')

        # loop through calibration images
        for image_path in calibration_images:
            # read each image
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            # add object points and image points (after refining them)
            if ret:
                objpoints.append(objp)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # draw and display the corners
                img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
                cv2.imwrite(self.path+'lab8_pkg/output_image.png', img)

        # perform camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx, dist


    def calculate_distance_to_cone(self, cone_image, reference_point):
        # read the cone image
        img = cv2.imread(cone_image)

        # set lower right corner pixel points
        # reference_point = [598, 418] # unknown
         # known

        # Calculate distance in x_car and y_car coordinates
        x_car = reference_point[0] * (self.mount_height / self.calibration_matrix[0, 0])
        y_car = reference_point[1] * (self.mount_height / self.calibration_matrix[1, 1])

        # Draw reference point for visualization
        cv2.circle(img, tuple(reference_point), 5, (0, 0, 255), -1)

        # Display visualizations
        cv2.imwrite(
            self.path+'lab8_pkg/cone_image_result.png', img)

        return x_car, y_car

    def calculate_camera_mount_height(self):
        pixel_coordinates = np.array([665, 499])  # Reshaped to (1, 2) array
        distance_to_object = 40

        homogeneous_coordinates = np.append(pixel_coordinates, 1)

        # Use inverse camera matrix to convert to camera frame
        inverse_camera_matrix = np.linalg.inv(self.calibration_matrix)
        point_in_camera_frame = np.dot(
            inverse_camera_matrix, homogeneous_coordinates)

        # Scale the point based on the known distance to object
        point_in_camera_frame *= distance_to_object

        # Assuming camera is at the origin, the height of the camera mount is the z-coordinate
        self.mount_height = point_in_camera_frame[2]

        self.get_logger().info(
            "Height of the camera mount: {:.2f} cm".format(self.mount_height))

    def main(self):

        self.calculate_camera_mount_height()

        # Calculate distance to cone
        cone_40_image = self.path+'lab8_pkg/resource/cone_x40cm.png'
        reference_point = [664, 494]
        x_car, y_car = self.calculate_distance_to_cone(cone_40_image, reference_point)

        if x_car is not None and y_car is not None:
            print("Distance to cone (x_car, y_car):", x_car, "cm,", y_car, "cm")
        else:
            print("Error: Distance calculation failed.")

        cone_image = self.path+'lab8_pkg/resource/cone_unknown.png'
        reference_point = [598, 418]
        x_car, y_car = self.calculate_distance_to_cone(
            cone_image, reference_point)

        if x_car is not None and y_car is not None:
            print("Distance to cone (x_car, y_car):", x_car, "cm,", y_car, "cm")
        else:
            print("Error: Distance calculation failed.")


def main(args=None):
    rclpy.init(args=args)
    distance_node = DistanceNode()
    distance_node.main()
    rclpy.spin(distance_node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()