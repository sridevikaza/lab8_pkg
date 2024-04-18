#!/usr/bin/env python3

from ultralytics import YOLO
from PIL import Image

from distance import calculate_camera_matrix,pixel_to_camera_coordinates, camera_to_car_coordinates, get_distance_to_cone

def calculate_mounting_height(pixel, camera_intrinsics):
    camera_coords = pixel_to_camera_coordinates(pixel, camera_intrinsics)

    height = camera_to_car_coordinates(camera_coords, [0.6, 0, 0])
    print("Height of camera:", height)
    return height


if __name__ == "__main__":
    images_folder = '/root/sim_ws/src/lab8_pkg/calibration'
    chessboard_size = (6, 8)
    square_size = 0.25  # 25cm
    camera_intrinsics = calculate_camera_matrix(
        images_folder, chessboard_size, square_size)

    pixel = (530, 311)
    camera_mounting_height = calculate_mounting_height(
        pixel, camera_intrinsics)
    print("Camera mounting height:", camera_mounting_height, "meters")


    # yolo inference
    model = YOLO('/root/sim_ws/src/lab8_pkg/yolo/best.pt')

    images = ['/root/sim_ws/src/lab8_pkg/yolo/img/69.jpg']
    # source_folder = '/root/sim_ws/src/lab8_pkg/yolo/img'
    # images = glob.glob(source_folder + '/*.jpg')

    for i, image in enumerate(images):
        r = model(image)[0]

        # Get bounding box coordinates
        x1, y1, x2, y2 = r.boxes.xyxy[0]

        # Calculate center pixel coordinates
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        print(f"Center pixel of bounding box {i+1}: ({center_x}, {center_y})")

        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

        # Save results to disk
        r.save(
            filename=f'/root/sim_ws/src/lab8_pkg/submission/yolo_result_distance.jpg')

        pixel = [center_x, center_y]
        print("Pixel coordinates:", pixel)

        x_car, y_car = get_distance_to_cone(
            pixel, camera_intrinsics, camera_mounting_height)
        print("Distance to car: x: ", x_car, " y: ", y_car)
