from cv2 import imread, cvtColor, GaussianBlur, Canny, findContours,drawContours
from cv2 import contourArea, arcLength, approxPolyDP, threshold, adaptiveThreshold
from cv2 import COLOR_BGR2GRAY, COLOR_BGR2RGB, CHAIN_APPROX_SIMPLE, RETR_LIST, THRESH_BINARY, ADAPTIVE_THRESH_GAUSSIAN_C, ADAPTIVE_THRESH_MEAN_C
from cv2 import imwrite
from imutils.perspective import four_point_transform
import matplotlib.pyplot as plt
import numpy as np
import os

import cv2

def order_points(pts):    
    a = np.sum(np.square(pts[0,0]-pts[1,0]))
    b = np.sum(np.square(pts[0,0]-pts[3,0]))

    if a > b:
        rect = pts.copy()
        rect[0] = pts[0]
        rect[1] = pts[3]
        rect[2] = pts[2]
        rect[3] = pts[1]
        return rect
    else:
        rect = pts.copy()
        rect[0] = pts[1]
        rect[1] = pts[0]
        rect[2] = pts[3]
        rect[3] = pts[2]
        return rect

def main(imgname):
    image = imread("images/documents/"+imgname+".jpg")

    imwrite("images/output/"+imgname+"_1_original.jpg", image)

    grayscale_image = cvtColor(image,COLOR_BGR2GRAY)
    blurred_image = GaussianBlur(grayscale_image, ksize=(5,5),sigmaX=0,sigmaY=0)

    imwrite("images/output/"+imgname+"_2_grayscaleimg.jpg", grayscale_image)
    imwrite("images/output/"+imgname+"_3_gaussianblur.jpg", blurred_image)

    edges = Canny(blurred_image,75,180, L2gradient=True)

    imwrite("images/output/"+imgname+"_4_canny.jpg", edges)

    contours, hierarchy = findContours(edges,RETR_LIST, CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=contourArea, reverse=True)

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)

        if len(approx) == 4:
            doc_cnts = approx
            break

    contoured_image = drawContours(image.copy(), doc_cnts, -1, (255,0,0), thickness=50)

    imwrite("images/output/"+imgname+"_5_contouredimage.jpg", contoured_image)

    # transformation = four_point_transform(image, doc_cnts.reshape(4, 2))
    # transformation = cvtColor(transformation, COLOR_BGR2GRAY)

    h = 3000
    w = int(np.floor(h*(1/np.sqrt(2))))
    dst_pts = np.array([[0, 0],   [w-1, 0],  [w-1, h-1], [0, h-1]], dtype=np.float32)
    src_pts = np.array(order_points(doc_cnts), dtype=np.float32).reshape(4,2)


    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    transformation = cv2.warpPerspective(grayscale_image, M, (w, h))

    imwrite("images/output/"+imgname+"_6_transformation.jpg", transformation)

    adaptive_binary_image_gaus = adaptiveThreshold(transformation,255,ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,11,6)
    adaptive_binary_image_mean = adaptiveThreshold(transformation,255,ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,7,6)
    tresh, binary_image = threshold(transformation, 100,255,THRESH_BINARY)

    imwrite("images/output/"+imgname+"_7_adaptive_binary_image_gaus.jpg", adaptive_binary_image_gaus)
    imwrite("images/output/"+imgname+"_8_adaptive_binary_image_mean.jpg", adaptive_binary_image_mean)
    imwrite("images/output/"+imgname+"_9_binary_image.jpg", binary_image)



if __name__ == "__main__":
    for i in os.listdir("images/documents/"):
        print(i.split(".")[0])
        main(i.split(".")[0])
        # exit()
