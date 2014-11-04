# -*- coding: utf-8 -*-
"""

@author: Julian Morris, Pratool Gadtaula

used by Adela Wee and Michelle Sit to debug issues with OpenCV face detection
"""

import scipy
import numpy
import cv2
import sys

#Create xml classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detectFaces():

    #reads in the images from the camera
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        if len(faces) != 0:
            print faces
            print "I see you!"

            # Draw a rectangle around the faces
            for (x,y,w,h) in faces:
                #terms of cv2 drawing shapes
                #referencing image where you want to draw shapes
                #center of shape
                #rectangle dimensions
                #color
                #line thickness
                #cv2.rectangle (img, vertex1, vertex across from 1, color, something)
                faceRect = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),0)
                # I don't know how/why this next line works
                # global roi_gray
                roi_gray = gray[y:y+h, x:x+w]
                #roi_color = frame[y:y+h, x:x+w]
                #print "here"
                #code to run smile detection (since smiles are on faces)
                smiles = smile_cascade.detectMultiScale(
                    roi_gray, 
                    scaleFactor=1.1, 
                    minNeighbors=10, 
                    minSize=(20,20),
                    maxSize=(80, 80),
                    flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
                print smiles
                print "smiles detected!"
                for (sx,sy,sw,sh) in smiles:
                    cv2.rectangle(frame,(sx+x,sy+y),(x+sw,y+sh),(0,0,255),0)
                    # global roi_gray2
                    roi_gray2 = gray[y+sy:y+sh, x+sx:x+sw]

                # Display the resulting frame
                import pdb
                pdb.set_trace()
                
                cv2.imshow('Video', frame)
                cv2.imshow('ROI', roi_gray)
                cv2.imshow('Other ROI', roi_gray2)
        else:
            print "There are no faces."

    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detectFaces()
