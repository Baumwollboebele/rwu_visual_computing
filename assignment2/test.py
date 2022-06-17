
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


def test_calibration():

    with open(path_summary_results + 'objs.pkl', 'rb') as f:
        t_error, ret, mtx, roi, newcameramtx, rvecs, tvecs, dist = pickle.load(f)
        print(t_error)
        print(ret)



if __name__ == '__main__':
    test_calibration()