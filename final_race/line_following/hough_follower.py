#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from std_msgs.msg import Float32
from vs_msgs.msg import ConeLocationPixel

# import your color segmentation algorithm; call this function in ros_image_callback!
from final_race.line_following.hough import cd_color_segmentation

lookahead=4.2

cut=150

PTS_IMAGE_PLANE = [[180, 317],
                   [416, 352],
                   [583, 385],
                   [299, 262],
                   [447, 245],
                   [203, 224],
                   [373, 218],
                   [87, 247],
                   [432, 276],
                   [236, 243],
                   [161, 282]] # dummy points
######################################################

# PTS_GROUND_PLANE units are in inches
# car looks along positive x axis with positive y axis to left

######################################################
## DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
PTS_GROUND_PLANE = [[18, 8.],
                    [15, -4.25],
                    [23.25, -17],
                    [27, 1.5],
                    [32.25, -12.],
                    [39.25, 12.75],
                    [43.5, -7.],
                    [31., 21.],
                    [24.25, -8.],
                    [31.75, 7.5],
                    [22.5, 10.25]] # dummy points
######################################################

METERS_PER_INCH = 0.0254

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
        self.debug_pub = self.create_publisher(Image, "/thresholded_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images
        self.lookahead_point_y = self.create_publisher(Float32, "/lookahead_point_y", 10)

        np_pts_ground = np.array(PTS_GROUND_PLANE)
        np_pts_ground = np_pts_ground * METERS_PER_INCH
        np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

        np_pts_image = np.array(PTS_IMAGE_PLANE)
        np_pts_image = np_pts_image * 1.0
        np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

        self.h, err = cv2.findHomography(np_pts_image, np_pts_ground)

        self.get_logger().info("Hough Follower Initialized")

    def transformUvToXy(self, u, v):
        """
        u and v are pixel coordinates.
        The top left pixel is the origin, u axis increases to right, and v axis
        increases down.

        Returns a normal non-np 1x2 matrix of xy displacement vector from the
        camera to the point on the ground plane.
        Camera points along positive x axis and y axis increases to the left of
        the camera.

        Units are in meters.
        """
        homogeneous_point = np.array([[u], [v+cut], [1]])
        xy = np.dot(self.h, homogeneous_point)
        scaling_factor = 1.0 / xy[2, 0]
        homogeneous_xy = xy * scaling_factor
        x = homogeneous_xy[0, 0]
        y = homogeneous_xy[1, 0]
        return x, y
        

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
        #image = image[100:, :, :]


        # image = "/home/racecar/racecar_ws/src/final_race/racetrack_images/lane_1/image1.png"
        lines,thresholded_image = cd_color_segmentation(image)

        thresholded_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)

        lookahead_points=[]

        mn=1e9

        right_lane=None

        if lines is not None:

            # k=2
            # kmeans = KMeans(n_clusters=k)
            # kmeans.fit(rt)

            # centers = kmeans.cluster_centers_

            # uvs=np.empty((0,2))

            for r_theta in lines:
                
                arr = np.array(r_theta[0], dtype=np.float64)
                r, theta = arr
                u=r*np.cos(theta)
                v=r*np.sin(theta)
                su=-v
                sv=u
                # if uvs.shape[0]>0 and np.min((u-uvs[:,0])**2+(v-uvs[:,1])**2)<=2500:
                #    continue
                # uvs=np.vstack((uvs,np.array([u,v])))
                x1,y1=self.transformUvToXy(u+(250-cut-v)*su/sv,250-cut)
                x2,y2=self.transformUvToXy(u+(200-cut-v)*su/sv,200-cut)
                sx=x2-x1
                sy=y2-y1
                sp1=np.array([x1,y1])
                d=np.array([sx,sy])
                qa=d.dot(d)
                qb=2*sp1.dot(d)

                t=-qb/(2*qa)
                cx=x1+t*sx
                cy=y1+t*sy
                angle=np.arctan2(cy,cx)

                #print(angle)

                if np.abs(angle-np.pi/2)<mn and cx**2+cy**2<1:
                    if right_lane!=None:
                        uu=right_lane[0]
                        vv=right_lane[1]
                        x1 = int(uu + 100*(-vv))
                        y1 = int(vv + 100*(uu))
                        x2 = int(uu - 100*(-vv))
                        y2 = int(vv - 100*(uu))
                        cv2.line(thresholded_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    mn=np.abs(angle-np.pi/2)
                    right_lane=(u,v)
                else:
                    x1 = int(u + 100*(-v))
                    y1 = int(v + 100*(u))
                    x2 = int(u - 100*(-v))
                    y2 = int(v - 100*(u))
                    cv2.line(thresholded_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if right_lane!=None:
                u=right_lane[0]
                v=right_lane[1]
                su=-v
                sv=u
                # print(u, v)
                # cv2.circle(thresholded_image, (int(u+((200-v)*su/sv)),200), 10, (255,0,0),5)
                # cv2.circle(thresholded_image, (int(u+((250-v)*su/sv)),250), 10, (0,0,255),5)

                x1,y1=self.transformUvToXy(u+(250-cut-v)*su/sv,250-cut)

                x2,y2=self.transformUvToXy(u+(200-cut-v)*su/sv,200-cut)
                # print(x1, y1)
                sp1=np.array([x1,y1])
                sx=x2-x1
                sy=y2-y1
                d=np.array([sx,sy])
                qa=d.dot(d)
                qb=2*sp1.dot(d)

                t=-qb/(2*qa)
                cx=x1+t*sx
                cy=y1+t*sy
                angle=np.arctan2(cy,cx)
                s=np.sign(cy)
                # print(cx,cy,angle)

                px1 = int(u + 100*(-v))
                py1 = int(v + 100*(u))
                px2 = int(u - 100*(-v))
                py2 = int(v - 100*(u))
                cv2.line(thresholded_image, (px1, py1), (px2, py2), (0, 255, 0), 2)

                # print(x,y,sx,sy)
                sp1=np.array([x1,y1-s*0.4])
                d=np.array([sx,sy])
                qa=d.dot(d)
                qb=2*sp1.dot(d)
                qc=sp1.dot(sp1)-lookahead**2

                if qb**2-4*qa*qc>=0:

                    t=(-qb+np.sqrt(qb**2-4*qa*qc))/(2*qa)
                    lx=x1+t*sx
                    ly=y1-s*0.4+t*sy

                    if s<0:
                        lx=0.0
                        ly=lookahead

                    lookahead_point = Point()
                    lookahead_point.x = lx
                    lookahead_point.y = ly
                    msg = Float32()
                    msg.data = ly
                    
                    self.lookahead_point_y.publish(msg)
                    self.lookahead_point_pub.publish(lookahead_point)



        debug_msg = self.bridge.cv2_to_imgmsg(thresholded_image, "bgr8")
        debug_msg.header.frame_id = "/camera_info"
        debug_msg.header.stamp = self.get_clock().now().to_msg()
        self.debug_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    cone_detector = LineDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()