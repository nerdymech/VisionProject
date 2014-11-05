# -*- coding: utf-8 -*-
"""

@author: Julian Morris, Pratool Gadtaula

used by Adela Wee and Michelle Sit to debug issues with OpenCV face detection
"""

import scipy
import numpy
import cv2
import sys

#in order to run file, type into terminal
#python webcam.py haarcascade_frontalface_default.xml

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

# mouthCascPath = sys.argv[2]
# mouthCascade = cv2.CascadeClassifier(mouthCascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = img[y:y+h, x:x+w]
        # mouth = mouthCascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
        import pdb
        pdb.set_trace()
    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
