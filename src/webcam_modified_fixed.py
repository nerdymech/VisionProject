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

#Create xml classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

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
            facedetectTime = datetime.now()
            print "face detected time: %s" %facedetectTime
            # Draw a rectangle around the faces
            for (x,y,w,h) in faces:
                print "I see %s people" %len(faces)
                #center of shape, rectangle dimensions, color, line thickness
                #cv2.rectangle (img, vertex1, vertex across from 1, color, something)
                faceRect = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),0)
                # print y
                # print y+h
                # print x
                # print x+w
                #need to readjust the crop to reflect proportional things
                roi_gray = gray[6*(y/5):y+(0.9*h),x+20:x+w-20]
                # roi_gray = gray[y+40:y+h-10,x+20:x+w-20]
                
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
                    if (smile_prob < -1):
                        print "no smile detected"
    
                    elif (smile_prob >= -1):
                        print "smile detected!"
                        smiledetectTime = datetime.now()
                        print "smile time: %s" %smiledetectTime
                        lag = smiledetectTime - facedetectTime
                        print "lag: %s" %lag
                    
                # print roi_gray
                # x_upperbound = x+ 2*(w/3)
                # cropped_face = roi_gray[y_upperbound:y+h, x_upperbound:x+w]
                # print cropped_face
                #roi_color = frame[y:y+h, x:x+w]
                #print "here"
                #code to run smile detection (since smiles are on faces)
                # smiles = smile_cascade.detectMultiScale(
                #     roi_gray, 
                #     scaleFactor=1.1, 
                #     minNeighbors=10, 
                #     minSize=(20,20),
                #     maxSize=(80, 80))
                # print "i found %s smiles" %len(smiles)
               
                # for (sx,sy,sw,sh) in smiles:
                #     cv2.rectangle(frame,(sx+x,sy+y),(x+sw,y+sh),(0,0,255),0)
                #     # global roi_gray2
                #     roi_gray2 = gray[y+sy:y+sh, x+sx:x+sw]

                    # import pdb
                    # pdb.set_trace()
            cv2.imshow("ROI", roi_gray)
            cv2.imshow('ROI_resized', resized_roi)
            # cv2.imshow('Cropped face', cropped_face)

                # Display the resulting frame
            
        cv2.imshow('Video', frame)
        # cv2.imshow('ROI', roi_gray)
        # cv2.imshow('Cropped face', cropped_face)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
        
        # import pdb
        # pdb.set_trace()

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model = train_smiles()
    detectFaces()
