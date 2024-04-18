#!/usr/bin/env python3

import cv2
import numpy as np
import glob
from ultralytics import YOLO
from PIL import Image
from distance import get_distance_to_cone, get_camera_matrix_and_height

# Main function
if __name__ == "__main__":
    
    # Camera intrinsics
    images_folder = '/root/sim_ws/src/lab8_pkg/calibration'
    camera_intrinsics, camera_mounting_height = get_camera_matrix_and_height(
        images_folder, (6, 8), 0.25)
    
    # yolo inference
    model = YOLO('/root/sim_ws/src/lab8_pkg/yolo/best.pt')
    source = '/root/sim_ws/src/lab8_pkg/yolo/1573.jpg'
    results = model(source)

    # Iterate through each detection
    for i, r in enumerate(results):
        # Get bounding box coordinates
        x1, y1, x2, y2 = r.boxes.xyxy[0]

        # Calculate center pixel coordinates
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        print(f"Center pixel of bounding box {i+1}: ({center_x}, {center_y})")

        # Visualize the results
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

        # Save results to disk
        r.save(filename=f'/root/sim_ws/src/lab8_pkg/yolo/yolo_result_{i}.jpg')

    print(results)

    pixel = [593, 413]
    # pixel = [center_x, center_y]
    print("Pixel coordinates:", pixel)

    x_car, y_car = get_distance_to_cone(
        pixel, camera_intrinsics, camera_mounting_height)
    print("Distance to cone in x_car:", x_car)
    print("Distance to cone in y_car:", y_car)
