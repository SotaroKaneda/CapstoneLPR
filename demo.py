import torch
import glob
import os
import cv2
import sys
import math
import time
import numpy as np
from utility import crop_from_points

tic = time.time()
lp_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', 'v-lp-detect-best2.pt')
toc = time.time()
lp_load_time = toc - tic

tic = time.time()
char_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', 'v-char-detect-best1.pt')
toc = time.time()
char_load_time = toc - tic

os.system("cls")

model = torch.hub.load('ultralytics/yolov5', 'custom', "best_weights/v-lp-detect-best.p")

# set model for inference
model.eval()    
# run image(s) through the model
results = model(image_location)

# results.xyxy has model predictions for each image given to the model
for prediction, image in zip(results.xyxy, image_location):
    # no bounding boxes were found in this image
    if prediction.numel() == 0:
        continue

    prediction = prediction.tolist()        # prediction is a pytorch tensor, need it as a list
    boxes = get_bounding_box_data(prediction, 0)

    for box in boxes:
        print(box)


# This saves the YOLOv5 output images to the the runs/detect directory
results.save()

print(f"LP Detect Load time: {lp_load_time} seconds.")
print(f"Character Detect Load time: {char_load_time} seconds.")

