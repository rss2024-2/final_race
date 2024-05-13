import rclpy
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive,AckermannDriveStamped
from geometry_msgs.msg import PoseArray,Point
from rclpy.node import Node
from visualization_msgs.msg import Marker


from tf_transformations import euler_from_quaternion

import numpy as np


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("pure_pursuit")
        self.declare_parameter('drive_topic', "/vesc/low_level/input/navigation")
        self.declare_parameter('lookahead_distance', "default")

        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.lookahead = 4.2
        self.speed = 4.0
        self.wheelbase_length = 0.3

        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                              self.drive_topic,
                                               1)
        self.lookahead_sub = self.create_subscription(Point,"/lookahead_point",self.lookahead_callback,1)

    def lookahead_callback(self,msg):

        eta=np.arctan2(msg.y,msg.x)
        delta=np.arctan2(2*self.lookahead*np.sin(eta),self.wheelbase_length)
        delta = np.clip(delta, -np.pi/47.5, np.pi/400) #first number is right autonomy, second is left autonomoy (-np.pi/50, 0) start 4 m/s lookahead 6
        
        #erica note for the johnson morningers, we need a little more left autonomy to make the turns, try pi/360 or pi/180; maybe np.pi/55? small drift


        header=Header()
        header.stamp=self.get_clock().now().to_msg()
        header.frame_id="base_link"
        drive=AckermannDrive()
        drive.steering_angle=delta
        drive.steering_angle_velocity=0.0
        drive.speed=self.speed
        drive.acceleration=0.0
        drive.jerk=0.0
        stamped_msg=AckermannDriveStamped()
        stamped_msg.header=header
        stamped_msg.drive=drive
        self.drive_pub.publish(stamped_msg)

def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()