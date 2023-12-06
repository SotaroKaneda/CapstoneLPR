# YOLO Setup: https://medium.com/analytics-vidhya/training-a-custom-object-detection-model-with-yolo-v5-aa9974c07088
# YOLO Custom Training: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#local-logging
# YOLO Object Detection Tutorial: https://docs.ultralytics.com/datasets/detect/
# Inference using custom trained model: https://github.com/ultralytics/yolov5/issues/7044

import torch
import glob
import os
import sys
from easyocr_test import easyocr_test
from utility import get_bounding_box_data


"""
    This script accepts either a single image or a folder of images to run YOLOv5 object detection on.
    argument 1: an image file name or image folder name
    argument 2: whether to run license plate(lp) detection or character detection(char)
        options: lp, char
"""


if len(sys.argv) < 3:
    print("Error. Supply image or image folder and a detection mode(lp for license plate, char for character)")
    sys.exit()

image_location = ""
weights = ""

# is the input a file or directory?
if os.path.isdir(sys.argv[1]):
    image_location = glob.glob(os.path.join(sys.argv[1], '*.*'))
else:
    image_location = sys.argv[1]

# load weights
if sys.argv[2] == "lp":
    weights = os.path.join("best_weights", "v-lp-detect-best.pt")
elif sys.argv[2] == "char":
    weights = os.path.join("best_weights", "v-char-detect-best.pt")
else:
    print("Error. Incorrect detection mode specified.")
    sys.exit()


model = torch.hub.load('ultralytics/yolov5', 'custom', weights)

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