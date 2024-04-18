#!/usr/bin/env python3

import cv2
import numpy as np
import glob

# Global variables to store clicked coordinates
clicked_x = -1
clicked_y = -1
clicked = False

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global clicked_x, clicked_y, clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_x = x
        clicked_y = y
        clicked = True
        print("Clicked at:", x, y)


# Function to calculate camera intrinsic matrix
def calculate_camera_matrix(images_folder, chessboard_size, square_size):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0],
                           0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(images_folder + '/*.png')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera intrinsic matrix:\n", mtx)
    return mtx


def pixel_to_camera_coordinates(pixel_coords, camera_intrinsics):

    # Convert pixel coordinates to homogeneous coordinates
    pixel_homogeneous = np.array([pixel_coords[0], pixel_coords[1], 1])

    # Get the inverse of the camera intrinsic matrix
    inv_intrinsics = np.linalg.inv(camera_intrinsics)

    # Apply inverse intrinsic matrix to pixel coordinates
    camera_coords_homogeneous = inv_intrinsics @ pixel_homogeneous

    return camera_coords_homogeneous


def camera_to_car_coordinates(camera_coords, car_coords):
    # Rotate camera coordinates
    R = np.array([[0, -1, 0],
                  [0, 0, -1],
                  [1, 0, 0]])
    rotated_coords = R @ car_coords

    # Calculate translation factor
    translation_factor = rotated_coords[2]

    # Translate to car coordinates
    translated_coords = camera_coords - rotated_coords / translation_factor
    translated_coords = translated_coords * translation_factor

    # Extract height
    height = translated_coords[1]

    return height

# Function to calculate camera mounting height
def calculate_mounting_height(pixel, camera_intrinsics):
    # Convert pixel coordinates to camera coordinates
    camera_coords = pixel_to_camera_coordinates(pixel, camera_intrinsics)

    # Convert camera coordinates to car coordinates and get height
    height = camera_to_car_coordinates(camera_coords, [0.4, 0, 0])

    print("Height of camera:", height)

    return height

# Function to get distance from camera to point in car frame
def get_distance_to_cone(cone_pixel_coords, intrinsic_matrix, camera_height):
    # Extract pixel coordinates of the cone
    pixel_u, pixel_v = cone_pixel_coords

    # Convert pixel coordinates to camera coordinates
    camera_coords_homogeneous = np.linalg.inv(
        intrinsic_matrix) @ np.array([pixel_u, pixel_v, 1])

    # Rotation matrix to convert camera coordinates to car coordinates
    rotation_matrix = np.array([[0, -1, 0],
                                [0, 0, -1],
                                [1, 0, 0]])

    # Translation vector representing the camera's position relative to the car
    translation_vector = np.array([0, camera_height, 0])

    # Calculate the scaling factor to convert camera coordinates to car coordinates
    scaling_factor = camera_height / camera_coords_homogeneous[1]

    # Transform camera coordinates to car coordinates
    car_coords_homogeneous = np.linalg.inv(
        rotation_matrix) @ (scaling_factor * camera_coords_homogeneous - translation_vector)

    # Extract x and y coordinates in the car frame
    x_car, y_car = car_coords_homogeneous[0], car_coords_homogeneous[1]

    return x_car, y_car


def click_on_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Create a window and set mouse callback
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)

    while True:
        # Display the image
        cv2.imshow('image', image)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


# Main function
if __name__ == "__main__":
    images_folder = '/root/sim_ws/src/lab8_pkg/calibration'
    chessboard_size = (6, 8)
    square_size = 0.25  # 25cm
    camera_intrinsics = calculate_camera_matrix(
        images_folder, chessboard_size, square_size)

    cone_x = 0.4  # 40cm

    # click_on_image(image_path = './resource/cone_x40cm.png')
    # pixel = (clicked_x, clicked_y)
    pixel = (661, 494)
    # Calculate camera mounting height
    camera_mounting_height = calculate_mounting_height(
        pixel, camera_intrinsics)
    print("Camera mounting height:", camera_mounting_height, "meters")

    # click_on_image(image_path='./resource/cone_unknown.png')
    # pixel = [clicked_x, clicked_y]
    pixel = [593, 413]

    x_car, y_car = get_distance_to_cone(
        pixel, camera_intrinsics, camera_mounting_height)
    print("Distance to cone in x_car:", x_car)
    print("Distance to cone in y_car:", y_car)
