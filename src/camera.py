#!/usr/bin/env python
# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

"""
This file opens an RGB camera and publishes images via ROS. 
It uses OpenCV to capture from camera 0.
"""

from __future__ import print_function
import rospy
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ImageSensor

from sensor_msgs.msg import Image as Image_msg
import numpy as np
import cv2

# My imports
import pyrealsense2 as rs

# My globals for realsense 2
cam_name = 'realsense_d435'
topic = '/dope/{}'.format(cam_name)
pipeline = None 
profile = None

def start_realsense():
    global pipeline, profile
    # Sets capture parameters
    width, height, fps = 640, 480, 30

    # Sets up realsense pipeline and config
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)

    # Returns pipeline and profile
    return [pipeline, profile]
    
def publish_images(freq=30):
    rospy.init_node(cam_name, anonymous=True)
    images_out = rospy.Publisher(topic, Image_msg, queue_size=10)
    rate = rospy.Rate(freq)

    print ("Publishing images from camera {} to topic '{}'...".format(
            cam_name, 
            topic
            )
    )
    print ("Ctrl-C to stop")
    while not rospy.is_shutdown():
        # Gets color frame from realsense
        frames = pipeline.wait_for_frames()
        frame_color = frames.get_color_frame()
        frame = np.asanyarray(frame_color.get_data())

        if frames:
            msg_frame_edges = CvBridge().cv2_to_imgmsg(frame, "bgr8")
            images_out.publish(msg_frame_edges)

        rate.sleep()

if __name__ == "__main__":
    
    try :
        start_realsense()
        publish_images()
    except rospy.ROSInterruptException:
        pass

