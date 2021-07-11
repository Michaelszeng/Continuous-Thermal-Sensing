"""
Run using "thermal-face" Conda environment

This program uses a TensorflowLite model to inference a bounding box on a live video feed and draws a
circle around the brighest pixel on the face.

For more info about how TensorflowLite works, see this page: https://www.tensorflow.org/lite/guide/python
"""

import tflite_runtime.interpreter as tflite
import argparse
import time
import math
import cv2

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_file', default='thermal_face_automl_edge_fast.tflite', help='model file. Note: ValueError signifies path to file is not valid')
parser.add_argument('-c', '--camera', default='1', help='which cameraID to stream video from')
parser.add_argument('--input_mean', default=127.5, type=float, help='input_mean')
parser.add_argument('--input_std', default=127.5, type=float, help='input standard deviation')
parser.add_argument('--window', default=50, type=float, help='window size for ROI detection')
args = parser.parse_args()

# def find_hottest_region(top_left, bottom_right, image, window_size=args.window):
#     """
#     Loops though the image and finds the window with the highest average brightness
#
#     Params
#     - top_left: tuple with (x, y) of the top left of the bounding box
#     - bottom right: tuple with (x, y) of the bottom right of the bounding box
#     - image: image of face
#     - window_size: the side length of the window of the ROI
#
#     Returns
#     - hottest_avg_val: the average pixel value of the hottest region
#     - hottest_avg_val_px: the (x, y) of the top left corner of the hottest region's window
#
#     CURRENTLY NON-FUNCTIONAL
#     """
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #convert frame to greyscale
#     w = bottom_right[0] - top_left[0]
#     h = bottom_right[1] - top_left[1]
#     print(w)
#     print(h)
#     hottest_total_val = 0
#     hottest_avg_val_px = (0, 0)
#     for x in range(0, w-window_size, 20):
#         for y in range(0, int(h/2)-window_size, 20):   #only check the top half of the bounding box for the ROI
#             total_val = 0
#             for r in range(window_size):
#                 for c in range(window_size):
#                     total_val += image[top_left[0]+x+r][top_left[1]+y+c]
#                     # print(total_val)
#                     # print(str(r) + ", " + str(c))
#             if total_val > hottest_total_val:
#                 hottest_total_val = total_val
#                 hottest_avg_val_px = (top_left[0]+x, top_left[1]+y)
#     hottest_avg_val = hottest_total_val / (math.pow(window_size, 2))
#     # print(hottest_avg_val)
#     # print(hottest_avg_val_px)
#     return hottest_avg_val, hottest_avg_val_px

#Loading the model
interpreter = tflite.Interpreter(model_path=args.model_file)
interpreter.allocate_tensors()

#Getting the input and output details of the model
input_details = interpreter.get_input_details()
print("input_details: ")
print(input_details)
print()
output_details = interpreter.get_output_details()
print("output_details: ")
print(output_details)
print()

input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']
input_type = input_details[0]['dtype']
output_type = output_details[0]['dtype']
print("input shape: " + str(input_shape))
print("output shape: " + str(output_shape))
print("input type: " + str(input_type))
print("output type: " + str(output_type))

height = input_shape[1]
width = input_shape[2]

#getting a video feed from the camera
cameraID = int(args.camera)
videoCapture = cv2.VideoCapture(cameraID)
if videoCapture.isOpened():  #try to get the first frame
    rval, image = videoCapture.read()
else:
    print("video read failed")
    rval = False

prev_time = time.time()  #This is for determining the loop time
while rval:
    #Loop time calculation
    now = time.time()
    print("loop time: %s" % (now-prev_time))
    prev_time = time.time()

    rval, image = videoCapture.read()   #Load a new image from the video feed

    #Convert the image into a black/white image with 3 color channels
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #convert frame to greyscale
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    #resize the image to the model's desired input dimensions
    image = cv2.resize(image, (width, height))

    if not rval:
        print("video read failed")

    input_data = np.expand_dims(image, axis=0)
    # print("input_data.shape: " + str(input_data.shape))

    t0 = time.time()
    #Feeding the input data into the model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    #Getting the output data from the model
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #Results is a 500x4 array. There are 500 detections (many of them will be noise) and each detection has 4 values to define the bounding box
    results = np.squeeze(output_data)
    t1 = time.time()
    # print("model eval time: " + str(t1-t0))

    h, w, ch = image.shape
    #Calculate the coordinates of the bounding box
    y1 = int(results[0][0] * h)
    x1 = int(results[0][1] * w)
    y2 = int(results[0][2] * h)
    x2 = int(results[0][3] * w)

    #Plot the first bounding box (for 1 person in frame)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # hottest_val, hottest_val_px = find_hottest_region((x1, y1), (x2, y2), image)
    # cv2.rectangle(image, hottest_val_px, (hottest_val_px[0]+args.window, hottest_val_px[1]+args.window), (0, 0, 255), 2)

    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #convert frame to greyscale
    # grey_blurred = cv2.GaussianBlur(grey, (21, 21), 0)   #not necessary for functionality, but might improve hottest region identification consistency
    grey_cropped = grey[y1:y2, x1:x2]   #crop the image so only the bounding box is present
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(grey_cropped)  #maxLoc is the coordinates of the brightest pixel in the cropped image
    cv2.circle(image, (x1+1*maxLoc[0], y1+1*maxLoc[1]), 5, (0, 50, 255), 1) #drawing a circle where the brighest pixel in the image is

    # Uncomment this to draw a 2nd bounding box (if there are 2 people in frame)
    # y3 = int(results[1][0] * w)
    # x3 = int(results[1][1] * h)
    # y4 = int(results[1][2] * w)
    # x4 = int(results[1][3] * h)
    # cv2.rectangle(image, (x3, y3), (x4, y4), (0, 255, 0), 0)

    image = cv2.resize(image, (w*2, h*2))   #scaling the image up to make the visualization more clear
    cv2.imshow("image", image)  #show the image, bounding box, and circle

    key = cv2.waitKey(20)
    if key == 27:   #exit on ESC
        break
