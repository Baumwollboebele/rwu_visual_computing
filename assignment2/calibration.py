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
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob("E:\\Home\\GitHub\\rwu_visual_computing\\assignment2\\vid_images\\*.jpg")
count = 0
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (6,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        #cv.drawChessboardCorners(img, (5,5), corners2, ret)
        cv.imwrite("E:\\Home\\GitHub\\rwu_visual_computing\\assignment2\\used_images\\" + "frame%d.jpg" % count, img)
        count +=1

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv.imread("E:\\Home\\GitHub\\rwu_visual_computing\\assignment2\\used_images\\frame0.jpg")
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite("E:\\Home\\GitHub\\rwu_visual_computing\\assignment2\\result\\result.jpg", dst)


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )