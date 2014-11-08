# -*- coding: utf-8 -*-
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

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

#Create xml classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

class smileDetect:

    def listener(self):
        rospy.init_node('detectFaces', anonymous =True)
        self.detectFaces()
        self.videothing()
        rospy.spin()
    
    def __init__(self):
        self.image = None
        self.camera_listener = rospy.Subscriber("camera/image_raw",Image, self.detectFaces)
        self.image_pub = rospy.Publisher("image_topic_2",Image)
        self.bridge = CvBridge() 
        print "got here"
        self.detectFaces(msg)
        self.videothing()
        print "here also"   
        # print "initialized"

    def detectFaces(self, msg):
        # pdb.set_trace()
        #reads in the images from the camera
        # video_capture = cv2.VideoCapture(0)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError, e:
            print e
        self.image = numpy.asanyarray(cv_image)
        print "self.image assigned"
        cv2.imshow ("cv_image", self.image)
        #need this to be in the same method and in the same indent as the cv2.imshow line
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass

    def videothing(self):

        pdb.set_trace()
        # while True:
        # Capture frame-by-frame
        # ret, frame = video_capture.read()
        # ret, image = image
        #return pauses that section of the code until the other section of the code runs
        print self.image
        if (self.image == None):
            print "no image detected"
            pass

        else: 
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            cv2.imshow ("self.image gray", gray)
            print "okay"

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(30, 30)
            )

            if len(faces) == 0:
                print "There are no faces."

            else:
                facedetectTime = datetime.now()
                print "face detected time: %s" %facedetectTime
                # Draw a rectangle around the faces
                for (x,y,w,h) in faces:
                    print "I see %s people" %len(faces)
                    #center of shape, rectangle dimensions, color, line thickness
                    #cv2.rectangle (img, vertex1, vertex across from 1, color, something)
                    faceRect = cv2.rectangle(self.image,(x,y),(x+w,y+h),(255,0,0),0)
                    roi_gray = gray[6*(y/5):y+(0.9*h),x+20:x+w-20]
                    # print roi_gray
                    # roi_gray = gray[y+40:y+h-10,x+20:x+w-20]
                    if len(roi_gray) == 0:
                        pass
                    else:
                        resized_roi = cv2.resize(roi_gray, (24, 24)).T/255.0
                        # scipy.misc.imsave('outfile.jpg', resized_roi)
                        roi_vec = resized_roi.reshape((resized_roi.shape[0]*resized_roi.shape[1],1))
                        smile_prob = -model.predict_log_proba(roi_vec.T)[0][0]
                        print "smile prob: %s" %smile_prob

                        if (smile_prob < 1):
                            print "no smile detected"   

                        elif (smile_prob >= 1):
                            print "smile detected!"
                            smiledetectTime = datetime.now()
                            print "smile time: %s" %smiledetectTime
                            lag = smiledetectTime - facedetectTime
                            print "lag: %s" %lag

                    #     # pdb.set_trace()
                cv2.imshow("ROI", roi_gray)
                cv2.imshow('ROI_resized', resized_roi)

            # cv2.imshow('Video', self.image)
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass

        # pdb.set_trace()

    # When everything is done, release the capture
    # video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':

    model = train_smiles()
    de = smileDetect()
    de.listener()
    # de = smileDetect.videothing()
    # rospy.init_node('detectFaces', anonymous =True)
    # rospy.spin()
