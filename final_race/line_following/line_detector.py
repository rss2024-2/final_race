#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from vs_msgs.msg import ConeLocationPixel

# import your color segmentation algorithm; call this function in ros_image_callback!
from final_race.line_following.color_segmentation import cd_color_segmentation


class LineDetector(Node):
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    def __init__(self):
        super().__init__("line_detector")
        # toggle line follower vs cone parker
        self.LineFollower = True

        # Subscribe to ZED camera RGB frames
        self.lookahead_point_pub = self.create_publisher(Point, "/lookahead_point", 10)
        self.cone_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 10)
        self.debug_pub = self.create_publisher(Image, "/cone_debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

        self.get_logger().info("Line Detector Initialized")
        

    def image_callback(self, image_msg):
        # Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        # From your bounding box, take the center pixel on the bottom
        # (We know this pixel corresponds to a point on the ground plane)
        # publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        # convert it to the car frame.

        #################################
        # YOUR CODE HERE
        # detect the cone and publish its
        # pixel location in the image.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #################################

        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        y, x, rgb = image.shape
        
        #for johnson track, changes lookahead distance %1m=39 in = 200 
        image = image[190:210, : , :]
       # print(image.shape) ( , 640, 3)
        #image[4*y//5:y ,: ,: ] = 0


        # image = "/home/racecar/racecar_ws/src/final_race/racetrack_images/lane_1/image1.png"
        # bounding_box = cd_color_segmentation(image, None)[0]
        _, thresholded_image, lookahead = cd_color_segmentation(image, None)

        lookahead_point = Point()
        lookahead_point.x = lookahead[0]
        lookahead_point.y = lookahead[1]
        self.lookahead_point_pub.publish(lookahead_point)
        
        # white_pixel_indices = np.where(thresholded_image == 255)

        # # Calculate the average position of white pixels
        # if len(white_pixel_indices[0]) > 0:
        #     average_position = (int(np.mean(white_pixel_indices[1])), int(np.mean(white_pixel_indices[0])))
        #     print(average_position)
        # else:
        #     average_position = None
        
        # middle_x = (bounding_box[0][0] + bounding_box[1][0])/2
        # middle_y = (bounding_box[0][1] + bounding_box[1][1])/2
        # lower_y = bounding_box[1][1]
        # cone_msg = ConeLocationPixel()
        # cone_msg.u = middle_x
        # cone_msg.v = float(lower_y)
        # self.cone_pub.publish(cone_msg)

        # # image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        # image = cv2.rectangle(image, bounding_box[0],bounding_box[1], (0, 255, 0),2)
        # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # color = image_hsv[int(middle_y)][int(middle_x)]
        # image = cv2.putText(image, np.array2string(color), (bounding_box[0][0], bounding_box[0][1] - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # low_bound = np.array([0, 100, 100])
        # high_bound = np.array([10, 255, 255]) 
        # cone_box = cd_color_segmentation(image, None)[0]
        # if cone_box[0] != cone_box[1]:
        #     p1, p2 = cone_box
        #     cv2.rectangle(image, p1, p2, (0,255,0), 2)

        # # debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        # # self.debug_pub.publish(debug_msg)

        # cone_px = ConeLocationPixel()
        # cone_px.u = (p1[0] + p2[0])/2
        # cone_px.v = (p1[1] + p2[1])/2 #follow centroid — most likely to be on line (assume cone on ground there)
        # self.cone_pub.publish(cone_px)
    

        debug_msg = self.bridge.cv2_to_imgmsg(thresholded_image, "bgr8")
        debug_msg.header.frame_id = "/camera_info"
        debug_msg.header.stamp = self.get_clock().now().to_msg()
        self.debug_pub.publish(debug_msg)

        #cone_px = ConeLocationPixel()
        #cone_px.u = float(average_position[0])
        #cone_px.v = float(average_position[1])
        #self.cone_pub.publish(cone_px)

def main(args=None):
    rclpy.init(args=args)
    cone_detector = LineDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()