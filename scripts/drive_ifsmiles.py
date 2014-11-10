#!/usr/bin/env python
"""
Code from CompRobo
Adela Wee and Michelle Sit"""

import pdb
from datetime import datetime
from webcam_modified_fixed import detectFaces

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String

def move_neato():
	pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
	self.image_pub = rospy.Publisher("image_topic_2",Image)
	self.camera_listener = rospy.Subscriber("camera/image_raw",Image, self.convertingNumpy)
	rospy.init_node('teleop', anonymous=True)
	r = rospy.Rate(10) #run at 10 hz

	#if time, add in a dist_buffer = 
	while not rospy.is_shutdown():
		#move robot! this is a state machine
		if faces >= 1:

			if webcam.avg >= 1:
				#Someone's smiling, so drive forwards
				velocity_msg = Twist(Vector3((0.2), 0.0, 0.0), Vector3(0.0, 0.0, 0.0))
				print (":D Hi there!")

			elif webcam.avg <0.8:
				#No webcam.avg detected, so drive backwards
				velocity_msg = Twist(Vector3(-0.2, 0.0, 0.0), Vector3(0.0, 0.0, 0.0))
				print (":( i'll go away now....")

		else:
			#move forwards at 0.2 m/s
			velocity_msg = Twist(Vector3(0.0, 0.0, 0.0), Vector3(0.0, 0.0, 0.0))
			print ("is anyone here? waiting for command")

		pub.publish(velocity_msg)
		r.sleep()

if __name__ == '__main__':
	try:
		webcam = detectFaces()
		move_neato()
	except rospy.ROSInterruptException: pass
