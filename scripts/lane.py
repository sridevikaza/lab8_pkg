#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        self.lane_publisher = self.create_publisher(
            Image, 'lane_detection', 10)

        self.bridge = CvBridge()

    def detect_lanes(self, image_path):
        # Read the image
        cv_image = cv2.imread(image_path)

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=50, minLineLength=100, maxLineGap=10)

        # Draw detected lines on the original image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert the OpenCV image to a ROS 2 image message
        lane_detection_msg = self.bridge.cv2_to_imgmsg(
            cv_image, encoding='bgr8')

        # Publish lane detection results
        self.lane_publisher.publish(lane_detection_msg)


def main(args=None):
    rclpy.init(args=args)
    lane_detection_node = LaneDetectionNode()

    # Path to the image file
    image_path = '/root/sim_ws/src/lab8_pkg/imgs/lane_sample.png'

    # Perform lane detection on the image
    lane_detection_node.detect_lanes(image_path)

    # Wait for messages to be published
    rclpy.spin_once(lane_detection_node)

    # Clean up
    lane_detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
