#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import glob

class DistanceNode(Node):
    def __init__(self):
        super().__init__('distance_node')
        calibration_path = '/home/moody/f1tenth_ws/src/lab8/calibration'

        # Initialize ROS 2 publisher for publishing distance information
        self.distance_publisher = self.create_publisher(Image, 'cone_distance', 10)

        # Load camera calibration matrix
        self.calibration_matrix, self.dist_coeffs = self.calibrate_camera(calibration_path, (6, 8), 25)

        # Initialize CvBridge for converting OpenCV images to ROS 2 images
        self.bridge = CvBridge()

    def calibrate_camera(self, calibration_folder, chessboard_size, square_size):
        # Prepare object points
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

        # Arrays to store object points and image points from all the images
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane

        # Get the list of calibration images
        calibration_images = glob.glob(calibration_folder + '/*.png')

        # Iterate through calibration images
        for image_path in calibration_images:
            # Read each image
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

        # Perform camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx, dist

    def calculate_distance_to_cone(self, cone_image):
        # Read the cone image
        img = cv2.imread(cone_image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect cone corners
        ret, corners = cv2.findChessboardCorners(gray, (6, 8), None)

        if ret:
            # Assuming the lower right corner of the nearest red cone is the reference point
            reference_point = corners[0][0]

            # Calculate distance in x_car and y_car coordinates
            x_car = reference_point[0] * (self.mount_height / self.calibration_matrix[0, 0])
            y_car = reference_point[1] * (self.mount_height / self.calibration_matrix[1, 1])

            return x_car, y_car
        else:
            print("Error: Chessboard corners not found in the cone image.")
            return None, None

    def publish_distance(self, x_car, y_car):
        # Create ROS 2 image message
        image_msg = Image()
        # Fill the image message data with distance information
        # For example, you could use the pixel coordinates (x_car, y_car) as distance
        image_msg.data = bytearray([x_car, y_car])
        # Publish the image message
        self.distance_publisher.publish(image_msg)

    def main(self):
        # Calibration parameters
        chessboard_size = (6, 8)  # Chessboard size (inner corners)
        square_size = 25  # Size of each square in mm
        self.mount_height = 0  # Initialize mount height

        # Calculate mount height
        x_car_known = 40  # Known x_car distance of the cone in cm
        self.mount_height = x_car_known * self.calibration_matrix[0, 0] / 100

        # Calculate distance to cone
        cone_image = '/home/moody/f1tenth_ws/src/lab8/resource/cone_unknown.png'
        x_car, y_car = self.calculate_distance_to_cone(cone_image)

        if x_car is not None and y_car is not None:
            print("Distance to cone (x_car, y_car):", x_car, "cm,", y_car, "cm")
            self.publish_distance(x_car, y_car)
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