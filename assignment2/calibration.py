# Abstract (Felix)
# Introduction (Felix) (Mario macht images)
# Preprocesseing = Video2Frame, Einlesen, Grayscale (Mario)
# Corner Detection = komplette for-schleife (Mario)
# Camera Calibartion = calibrateCamera, getOptimalNewCameraMatrix (Mario)
# Undistortion = undistort, images, error (felix)
# Summary = Error und dMatrix nochmal darstellen (felix)

import numpy as np
import cv2 as cv
import glob
import os
import pickle

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
path_used_image = "used_images/"
path_result = "result/"
path_vid_images = "vid_images/"
path_summary_results = "summary_results/"
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def calibration():
    objp = np.zeros((6*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:6].T.reshape(-1,2)
    objpoints = [] 
    imgpoints = [] 

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    images = os.listdir(path_vid_images)
    images = [path_vid_images+i for i in images]
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    count = 0
    max_it = len(images)
    for i, fname in enumerate(images):
        print("iteration1: ", i, " of ", max_it)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (6,6), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv.imwrite(path_used_image + "frame%d.jpg" % count, img)
            count +=1

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    img = cv.imread(path_used_image + "frame0.jpg")
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(path_result + "result.jpg", dst)


    mean_error = 0
    max_it = len(objpoints)
    for i in range(len(objpoints)):
        print("iteration2: ", i, " of ", max_it)
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    t_error = mean_error/len(objpoints)

    with open(path_summary_results + 'objs.pkl', 'wb') as f:
        pickle.dump([t_error, ret, mtx, roi, newcameramtx, rvecs, tvecs, dist], f)

    print(t_error)

def test_calibration():

    with open(path_summary_results + 'objs.pkl', 'rb') as f:
        t_error, ret, mtx, roi, newcameramtx, rvecs, tvecs, dist = pickle.load(f)
        print(t_error)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        images = os.listdir(path_vid_images)
        images = [path_vid_images+i for i in images]
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        max_it = len(images)
        for i, image in enumerate(images):
            print(i, " of ", max_it)
            img = cv.imread(image)
            h,  w = img.shape[:2]
            dst = cv.undistort(img, mtx, dist, None, newcameramtx)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            if "/" in image:
                image = image.split("/")[-1]
                image = image.split(".")[1]
            else:
                image = image.split("\\")[-1]
                image = image.split(".")[1]


            cv.imwrite(path_result + image + "_result.jpg", dst)

if __name__ == '__main__':
    calibration()
    test_calibration()

