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
        cv_image = cv2.imread(image_path)

        yellow_mask = self.segment_yellow(cv_image)

        masked_image = cv2.bitwise_and(cv_image, cv_image, mask=yellow_mask)

        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=50, minLineLength=100, maxLineGap=10)

        if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        lane_detection_msg = self.bridge.cv2_to_imgmsg(
            cv_image, encoding='bgr8')

        self.lane_publisher.publish(lane_detection_msg)

    def segment_yellow(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        return yellow_mask
    
def main(args=None):
    rclpy.init(args=args)
    lane_detection_node = LaneDetectionNode()

    image_path = '/root/sim_ws/src/lab8_pkg/imgs/lane_sample.png'

    lane_detection_node.detect_lanes(image_path)

    rclpy.spin_once(lane_detection_node)

    lane_detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
