import numpy as np
import cv2

# Known parameters
# Correct shape and data type
pixel_coordinates = np.array([[665, 499]], dtype=np.float64)
distance_to_object = 40  # Distance from the camera to the object in cm

fx = 693.677
fy = 694.76
cx = 448
cy = 250
k1 = 0.17038642
k2 = -0.06207556
p1 = -0.00967646
p2 = -0.01192096
k3 = -1.02088799

# Camera calibration parameters
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float64)
distortion_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

# Undistort the pixel coordinates
undistorted_pixel_coordinates = cv2.undistortPoints(
    pixel_coordinates, camera_matrix, distortion_coeffs)

# Convert pixel coordinates to normalized image coordinates
normalized_image_coordinates = (
    undistorted_pixel_coordinates - np.array([[cx, cy]])) / np.array([[fx, fy]])

# Convert normalized image coordinates to homogeneous coordinates
homogeneous_coordinates = np.hstack(
    (normalized_image_coordinates, np.ones((1, 1), dtype=np.float64))).T

# Use inverse camera matrix to convert to camera frame
inverse_camera_matrix = np.linalg.inv(camera_matrix)
point_in_camera_frame = np.dot(inverse_camera_matrix, homogeneous_coordinates)

# Scale the point based on the known distance to object
point_in_camera_frame *= distance_to_object

# Assuming camera is at the origin, the height of the camera mount is the z-coordinate
height_of_camera_mount = point_in_camera_frame[2]

print("Height of the camera mount:", height_of_camera_mount)
