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
# path_used_image = "E:\\Home\\GitHub\\rwu_visual_computing\\assignment2\\used_images\\"
# path_result = "E:\\Home\\GitHub\\rwu_visual_computing\\assignment2\\result\\"
# path_vid_images = "E:\\Home\\GitHub\\rwu_visual_computing\\assignment2\\vid_images\\"
# path_summary_results = "E:\\Home\\GitHub\\rwu_visual_computing\\assignment2\\summary_results\\"

path_used_image = "used_images/"
path_result = "result/"
path_vid_images = "vid_images/"
path_summary_results = "summary_results/"
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def calibration():

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:6,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # images = glob.glob(path_vid_images + "*.jpg") 

    images = os.listdir(path_vid_images)
    images = [path_vid_images+i for i in images]
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    count = 0
    max_it = len(images)
    for i, fname in enumerate(images):
        print("iteration1: ", i, " of ", max_it)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (6,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            # corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria) # Commented out cause it was only for drawing
            imgpoints.append(corners)
            # Draw and display the corners
            #cv.drawChessboardCorners(img, (5,5), corners2, ret) # Commented out cause it was only for drawing
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

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # images = glob.glob(path_vid_images + "*.jpg") 

        images = os.listdir(path_vid_images)
        images = [path_vid_images+i for i in images]
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        max_it = len(images)
        for i, image in enumerate(images):
            print(i, " of ", max_it)
            img = cv.imread(image)
            h,  w = img.shape[:2]
            # newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
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

