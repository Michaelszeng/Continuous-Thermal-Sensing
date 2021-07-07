"""
Run using "thermal-face" Conda environment
"""

import tflite_runtime.interpreter as tflite
import argparse
import time
import cv2

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_file', default='thermal_face_automl_edge_fast.tflite', help='model file. Note: ValueError signifies path to file is not valid')
parser.add_argument('-i', '--image', default='image.jpg', help='image file.')
parser.add_argument('--input_mean', default=127.5, type=float, help='input_mean')
parser.add_argument('--input_std', default=127.5, type=float, help='input standard deviation')
args = parser.parse_args()

interpreter = tflite.Interpreter(model_path=args.model_file)
interpreter.allocate_tensors()

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
img = Image.open(args.image).convert('RGB').resize((width, height))

input_data = np.expand_dims(img, axis=0)
print("input_data.shape: " + str(input_data.shape))

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
results = np.squeeze(output_data)
print(results.shape)
print(results)

np.save("results.npy", results)

image = cv2.imread(args.image)
h, w, ch = image.shape
y1 = int(results[0][0] * w)
x1 = int(results[0][1] * h)
y2 = int(results[0][2] * w)
x2 = int(results[0][3] * h)
cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 0)
cv2.imshow("image", image)
cv2.waitKey(0)
