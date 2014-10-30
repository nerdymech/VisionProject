
import numpy as np
import cv2
import scipy
import sys


#Create xml classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

#read the image
img = cv2.imread('family.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Find faces in the image, returns positions of detected faces
#faces are returned as Rect(x,y,w,h)
#scale factor accounts for faces that are closer to the front that appear bigger than the ones in the back
#minNeighbors is a moving window that defines how many objects are detected near the current one before it declares the faces found
#minSize gives the size of each window.
#print "HELLO"
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=10,
    minSize=(40,40)
    )
#print "detected faces"

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    #print "here"
    #code to run smile detection (since smiles are on faces)
    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=4, minSize=(30,30))

    for (sx,sy,sw,sh) in smiles:
        cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
    #eyes = eye_cascade.detectMultiScale(roi_gray)
    #print "eye"
    #for (ex,ey,ew,eh) in eyes:
     #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#Display resultant frame
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
