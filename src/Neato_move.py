#!/usr/bin/env python

#Comprobo Vision Project
#Adela Wee and Michelle Sit
#Fall 2014

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Vector3
from cv_bridge import CvBridge,CvBridgeError
import cv2

#fill this in for the camera
#from sensor_msgs.msg import LaserScan

#init vars

def smile_received(msg,pub):
    #processes shots from Neato

    #convert neato ros img msgs into opencv images w cv_bridge

    #feed necessary coordinates from opencv into things that get published


def unmanned_drive():
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    #fill this in for the camera
    #sub = rospy.Subscriber('scan', LaserScan, scan_received, pub)
    r = rospy.Rate(10) #run at 10 hz

    while not rospy.is_shutdown():
        #move the robot
        #gains
        kp = 0.4
        #if person is to the left of the center of the image, move left
        if person_coordinates < center_of_image- 5pixels and smile_received == True:
            #recall that x,y,z forward velocity then x,y,z rotational velocity
            #should probably acct for proximity with laser scan
            velocity_msg = Twist(Vector3((0.1,0.0,0.0),Vector3(0.0,0.0,kp*location_C)))
            print "happniess detected, moving left"

        elif person coordinates > center of image -5pixels and smile_received == True:
            velocity_msg = Twist(Vector3((0.1,0.0,0.0),Vector3(0.0,0.0,kp*location_C)))
            print "happiness detected, moving right"

        elif smile_received == False:
            velocity_msg = Twist(Vector3(-0.1,0.0,0.0),Vector3(0.0,0.0,0.0)))
            print "i go away from unhappy people"
    else: 
        #stay put
        velocity_msg = Twist(Vector3(0.0,0.0,0.0),Vector3(0.0,0.0,0.0)))
        print "I don't see any faces!"

    pub.publish(velocity_msg)
    r.sleep()

if __name__ == '__main__':
    try:
        unmanned_drive()
    except rospy.ROSInterruptException: pass