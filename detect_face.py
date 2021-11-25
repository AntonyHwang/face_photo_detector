import cv2
import numpy as np
import rawpy

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('test.ARW')
raw_img = rawpy.imread('test.ARW')
img = raw_img.postprocess()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

scale_percent = 10 # percent of original size
width = int(gray.shape[1] * scale_percent / 100)
height = int(gray.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized_gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)

faces = face_cascade.detectMultiScale(resized_gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(resized_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('img', resized_gray)
    cv2.waitKey()