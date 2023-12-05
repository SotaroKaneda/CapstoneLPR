import torch
import glob
import os
import cv2
import sys
import math
import time
import numpy as np
from crop_LP import crop_from_points

tic = time.time()
lp_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', 'v-lp-detect-best2.pt')
toc = time.time()
lp_load_time = toc - tic

tic = time.time()
char_detect_model = torch.hub.load('ultralytics/yolov5', 'custom', 'v-char-detect-best1.pt')
toc = time.time()
char_load_time = toc - tic

os.system("cls")

print(f"LP Detect Load time: {lp_load_time} seconds.")
print(f"Character Detect Load time: {char_load_time} seconds.")

