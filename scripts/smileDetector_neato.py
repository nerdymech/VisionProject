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
from time import sleep

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

#Create xml classifier for OpenCV Haar Cascade 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class smileDetect:
    """Initialize nodes"""
    def __init__(self):
        #pdb.set_trace()
        self.image = None
        self.image_pub = rospy.Publisher("image_topic_2",Image)
        self.bridge = CvBridge() 
        self.camera_listener = rospy.Subscriber("camera/image_raw",Image, self.convertingNumpy)
        
        print "initialized"

    """create helper functions"""

    def convertingNumpy(self, msg):
        print "converting image"
        #reads in the images from the camera
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError, e:
            print e
        self.image = numpy.asanyarray(cv_image)
        #print "self.image has length %s" %len(self.image)
        #run facial detection algorithm
        self.detectFaces()
        cv2.imshow ("cv_image", self.image)
        #need this to be in the same method and in the same indent as the cv2.imshow line
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
    # def AvgValues(self):
    #     #code averages values in an array
    #     #create array to avg values from past 10 frames (~1 sec)
    #     print "Avg values called!"
    #     smileArr = []
    #     for x in range(10):
    #         smileArr.append(0)
    #     for y in range(0, len(smileArr)):
    #         "Avg values working!"
    #         smileArr.insert(y,self.smile_prob)
    #         try:
    #             avg = numpy.mean(smileArr)
    #             print "Avg Psmile %s" %avg
    #         except StatisticsError,e:
    #             print e
    #         if y == len(smileArr):
    #             y = 0

    """face and smile detection function"""
    def detectFaces(self):
        print "looking for faces"
        if (self.image == None):
            #print "no image detected"
            return
        else: 
            #sleep(0.5) ##for debugging purposes
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            cv2.imshow ("self.image gray", gray)
            #print "converted to greyscale!"

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(10, 10)
            )
            #print "the array faces is %s" %len(faces)
            if len(faces) == 0:
                print "There are no faces."

            else:
                facedetectTime = datetime.now()
                #print "face detected time: %s" %facedetectTime
                # Draw a rectangle around the faces
                for (x,y,w,h) in faces:
                    print "I see %s people" %len(faces)
                    #center of shape, rectangle dimensions, color, line thickness
                    #cv2.rectangle (img, vertex1, vertex across from 1, color, something)
                    faceRect = cv2.rectangle(self.image,(x,y),(x+w,y+h),(255,0,0),0)
                    roi_gray = gray[6*(y/5):y+(0.9*h),x+20:x+w-20]
                
                    if len(roi_gray) == 0:
                        pass
                    else:
                        resized_roi = cv2.resize(roi_gray, (24, 24)).T/255.0
                        # scipy.misc.imsave('outfile.jpg', resized_roi)
                        roi_vec = resized_roi.reshape((resized_roi.shape[0]*resized_roi.shape[1],1))
                        # smile_prob = -model.predict_log_proba(roi_vec.T)[0][0]
                        # self.AvgValues()
                        #print "smile prob: %s" %smile_prob
                        
                        if (smile_prob < 1):
                            print "no smile detected"   

                        elif (smile_prob >= 1):
                            print "smile detected!"
                            #neato camera ~ 10fps
                            smiledetectTime = datetime.now()
                            #print "smile time: %s" %smiledetectTime
                            lag = smiledetectTime - facedetectTime
                            print "lag: %s" %lag

                    #     # pdb.set_trace()
                cv2.imshow("ROI", roi_gray)
                cv2.imshow('ROI_resized', resized_roi)

            # cv2.imshow('Video', self.image)
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass

if __name__ == '__main__':
    #pdb.set_trace()
    #model = train_smiles()
    print "training smile detector!"
    rospy.init_node('convertingNumpy', anonymous =True)
    df = smileDetect()
    print "running Smile Detector!"

    try:
        print "rospy.spin!"
        rospy.spin()
    except KeyboardInterrupt:
        print "Quitting program"
        cv2.destroyAllWindows()
