# -*- coding: utf-8 -*-
"""

@author: Julian Morris, Pratool Gadtaula

used by Adela Wee and Michelle Sit to debug issues with OpenCV face detection
"""
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs
import scipy
import numpy
import cv2
import sys
import roslib

#Create xml classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

video_capture = cv2.VideoCapture(0)

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    print "faces detected!"

    # Draw a rectangle around the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = frame[y:y+h, x:x+w]
        #print "here"
        #code to run smile detection (since smiles are on faces)
        smiles = smile_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=4, maxSize=(30,20))
        print "smiles detected!"
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(frame,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
