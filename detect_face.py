import cv2
import numpy as np
import rawpy
import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile

def detect_face(face_cascade, eye_cascade, fn, test = False):
    raw_img = rawpy.imread(fn)
    img = raw_img.postprocess()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale_percent = 10 # percent of original size
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized_gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)

    faces = face_cascade.detectMultiScale(
        resized_gray, 
        scaleFactor = 1.1, 
        minNeighbors = 4
    )

    if test == True:
        for (x, y, w, h) in faces:
            cv2.rectangle(resized_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow('img', resized_gray)
            cv2.waitKey()
        return False

    if len(faces) == 0:
        return False
    
    eyes = face_cascade.detectMultiScale(
        resized_gray, 
        scaleFactor = 1.1, 
        minNeighbors = 4
    )

    if len(eyes) == 0:
        return False
        
    return True

if __name__ == '__main__' :
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    CHECK_GT = False

    FACE_GT_DIR = './ground_truth/face'
    NON_FACE_GT_DIR = './ground_truth/non_face'
    FACE_GT = [f for f in listdir(FACE_GT_DIR) if isfile(join(FACE_GT_DIR, f) )]
    NON_FACE_GT = [f for f in listdir(NON_FACE_GT_DIR) if isfile(join(NON_FACE_GT_DIR, f) )]

    if CHECK_GT == True:
        error = 0

    print('Input folder:')
    input_dir = str(input())
    print('Output folder:')
    output_dir = str(input())

    face_dir = os.path.join(output_dir, 'face')
    non_face_dir = os.path.join(output_dir, 'non_face')

    file_names = [f for f in listdir(input_dir) if isfile(join(input_dir, f) )]
    file_names = [f for f in file_names if f.endswith('.ARW')]

    if not os.path.exists(face_dir):
        os.makedirs(face_dir)

    if not os.path.exists(non_face_dir):
        os.makedirs(non_face_dir)

    total_file_num = len(file_names)
    file_count = 0

    for fn in file_names:
        file_count += 1
        f_path = os.path.join(input_dir, fn)

        if CHECK_GT == True:
            if detect_face(face_cascade, eye_cascade, f_path) == True:
                if fn not in FACE_GT:
                    error += 1
            else:
                if fn not in NON_FACE_GT:
                    error += 1
        else:
            if detect_face(face_cascade, eye_cascade, f_path) == True:
                copyfile(f_path, os.path.join(face_dir, fn))
            else:
                copyfile(f_path, os.path.join(non_face_dir, fn))
        
        print('Completed: ' + str(file_count) + '/' + str(total_file_num) + ' ' + str(round(file_count/total_file_num*100, 2)) + '%\n')
    
    if CHECK_GT == True:
        print('Correctness: ' + str(round(100 - error/total_file_num*100, 2)) + '%')

    # detect_face(face_cascade, eye_cascade, './DSC02168.ARW', True)
    # detect_face(face_cascade, eye_cascade, upperbody_cascade, './test0.ARW', True)
    
        


# img = cv2.imread('test.ARW')

