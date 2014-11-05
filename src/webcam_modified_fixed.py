# -*- coding: utf-8 -*-
"""

Code from CompRobo
Adela Wee and Michelle Sit"""

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
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            print "There are no faces."
            
        else:
            # Draw a rectangle around the faces
            for (x,y,w,h) in faces:
                print "I see %s people" %len(faces)
                #center of shape, rectangle dimensions, color, line thickness
                #cv2.rectangle (img, vertex1, vertex across from 1, color, something)
                faceRect = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),0)
                # I don't know how/why this next line works
                # global roi_gray
                roi_gray = gray[y:y+h, x:x+w]
                print roi_gray
                x_upperbound = x+ 2*(w/3)
                y_upperbound = y+ 2*(h/3) 
                print y_upperbound
                print y+h
                print x_upperbound
                print x+w
                cropped_face = roi_gray[y_upperbound:y+h, x_upperbound:x+w]
                print cropped_face
                #roi_color = frame[y:y+h, x:x+w]
                #print "here"
                #code to run smile detection (since smiles are on faces)
                smiles = smile_cascade.detectMultiScale(
                    roi_gray, 
                    scaleFactor=1.1, 
                    minNeighbors=10, 
                    minSize=(20,20),
                    maxSize=(80, 80))
                print "i found %s smiles" %len(smiles)
               
                for (sx,sy,sw,sh) in smiles:
                    cv2.rectangle(frame,(sx+x,sy+y),(x+sw,y+sh),(0,0,255),0)
                    # global roi_gray2
                    roi_gray2 = gray[y+sy:y+sh, x+sx:x+sw]

                    # import pdb
                    # pdb.set_trace()
            cv2.imshow('ROI', roi_gray)
            # cv2.imshow('Cropped face', cropped_face)

                # Display the resulting frame
            # else:
            #     print "There are no faces."
            
            cv2.imshow('Video', frame)
            # cv2.imshow('ROI', roi_gray)
            # cv2.imshow('Cropped face', cropped_face)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass
            
            import pdb
            pdb.set_trace()

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detectFaces()
