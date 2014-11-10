#!/usr/bin/env python
"""
Code from CompRobo
Adela Wee and Michelle Sit"""

import scipy
import numpy
import cv2
from train_smile import train_smiles
import sys
import pdb
from datetime import datetime
from cv2 import cv


import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String

#global faces, avg

#Create xml classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detectFaces():
    #reads in the images from the camera
    avg = 0
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(30, 30)
        )
        

    if faces:
        # Draw a rectangle around the faces
        for (x,y,w,h) in faces:
            print "I see %s people" %len(faces)
            #center of shape, rectangle dimensions, color, line thickness
            #cv2.rectangle (img, vertex1, vertex across from 1, color, something)
            faceRect = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),0)
            #need to readjust the crop to reflect proportional things
            roi_gray = gray[6*(y/5):y+(0.9*h),x+20:x+w-20]
            
            #resize roi_gray to (24, 24)
            if len(roi_gray) == 0:
                pass
            else:
                resized_roi = cv2.resize(roi_gray, (24, 24)).T/255.0
                # scipy.misc.imsave('outfile.jpg', resized_roi)
                roi_vec = resized_roi.reshape((resized_roi.shape[0]*resized_roi.shape[1],1))
                smile_prob = -model.predict_log_proba(roi_vec.T)[0][0]
                print "smile prob: %s" %smile_prob

                #the following still needs to be adjusted
                if (smile_prob < 0.7):
                    print "no smile detected"
                    move_neato(back)

                elif (smile_prob >= 1):
                    print "smile detected!"
                    smileArr.remove(smileArr[counter])
                    smileArr.insert(counter,smile_prob)
                    move_neato(forward)
        cv2.imshow("ROI", roi_gray)
        cv2.imshow('ROI_resized', resized_roi)
        # cv2.imshow('Cropped face', cropped_face)        
    
    elif len(faces) == 0:
        print "There are no faces."
        
    cv2.imshow('Video', frame)


def move_neato(command):
	pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
	
	r = rospy.Rate(100) #run at 10 hz

	#if time, add in a dist_buffer = 
	#move robot! this is kind of a state machine
	if command == "forward":
		velocity_msg = Twist(Vector3((0.2), 0.0, 0.0), Vector3(0.0, 0.0, 0.0))
		print (":D Hi there!")

	elif command == "back":
		velocity_msg = Twist(Vector3(-0.2, 0.0, 0.0), Vector3(0.0, 0.0, 0.0))
 		print (":( i'll go away now....")

	else:
		#stay there
		velocity_msg = Twist(Vector3(0.0, 0.0, 0.0), Vector3(0.0, 0.0, 0.0))
		print ("is anyone here? waiting for command")

	pub.publish(velocity_msg)
	r.sleep()

if __name__ == '__main__':
	try:
		rospy.init_node('teleop', anonymous=True)
		model = train_smiles()
        video_capture = cv2.VideoCapture(0)
		while not rospy.is_shutdown():
		  detectFaces()
        video_capture.release()
        cv2.destroyAllWindows()

		
	except rospy.ROSInterruptException: pass
