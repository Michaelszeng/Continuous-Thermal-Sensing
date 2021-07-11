"""
This was a test program that is no longer of use.
"""

import argparse
import time
import math
import cv2

import numpy as np
from PIL import Image

image_file = "test.jpg"
top_left = (170, 78)
bottom_right = (374, 478)

def find_hottest_region(top_left, bottom_right, image, window_size=50, stride=20):
    """
    Loops though the image and finds the window with the highest average brightness

    Params
    - top_left: tuple with (x, y) of the top left of the bounding box
    - bottom right: tuple with (x, y) of the bottom right of the bounding box
    - image: image of face
    - window_size: the side length of the window of the ROI

    Returns
    - hottest_avg_val: the average pixel value of the hottest region
    - hottest_avg_val_px: the (x, y) of the top left corner of the hottest region's window
    """
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #convert frame to greyscale
    w = bottom_right[0] - top_left[0]
    h = bottom_right[1] - top_left[1]
    print(w)
    print(h)
    hottest_avg_val = 0
    hottest_avg_val_px = (0, 0)
    for x in range(0, w-window_size, stride):
        for y in range(0, int(h/2)-window_size, stride):   #only check the top half of the bounding box for the ROI
            total_val = 0
            for r in range(window_size):
                for c in range(window_size):
                    total_val += image[top_left[0]+x+r][top_left[1]+y+c][0]
                    if image[top_left[0]+x+r][top_left[1]+y+c][0] < 0:
                        print("less than 0")
                    if image[top_left[0]+x+r][top_left[1]+y+c][0] > 255:
                        print("greater than 255")
                    # print(total_val)
                    # print(str(r) + ", " + str(c))
            avg_val = total_val / (math.pow(window_size, 2))
            # print(avg_val)
            if avg_val > hottest_avg_val:
                print("avg_val: " + str(avg_val))
                hottest_avg_val = avg_val
                hottest_avg_val_px = (top_left[0]+x, top_left[1]+y)
            cv2.rectangle(image, (top_left[0]+x, top_left[1]+y), (top_left[0]+x+window_size, top_left[1]+y+window_size), (0, 0, 255), 1)
            cv2.imshow("image", image)
            cv2.waitKey(0)
            key = cv2.waitKey(20)
            if key == 27:   #exit on ESC
                exit()

    # print(hottest_avg_val)
    # print(hottest_avg_val_px)
    return hottest_avg_val, hottest_avg_val_px

image = cv2.imread(image_file)
grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #convert frame to greyscale
cv2.rectangle(image, (170, 78), (374, 478), (255, 0, 0), 2)
# find_hottest_region(top_left, bottom_right, image, window_size=50)

# grey = cv2.GaussianBlur(grey, (65, 65), 0)
cv2.imshow("grey", grey)
cv2.waitKey(0)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grey)
cv2.circle(image, maxLoc, 25, (255, 0, 0), 2)
cv2.imshow("img", image)
cv2.waitKey(0)
