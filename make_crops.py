import torch
import glob
import os
import cv2
import sys
import math
import numpy as np
from crop_LP import crop_from_points


"""
    Crop License Plates or Characters off of a license plate
    arg1: image folder
    arg2: lp for license plate
          char for cropping charater off of a cropped license plate
"""

image_folder = ""
crop_folder = ""
char_crop_folder = "chars"
lp_crop_folder = "lp_crops"


# TODO add command line checking here
if sys.argv[2] == "lp":
    crop_folder = lp_crop_folder
elif sys.argv[2] == "char":
    crop_folder = char_crop_folder


if len(sys.argv) < 2:
    image_folder = "car_images"
else:
    image_folder = sys.argv[1]

model = torch.hub.load('ultralytics/yolov5', 'custom', 'v-char-detect-best1.pt')

# load trained weights
# model.load_state_dict(torch.load('yolov5/runs/train/exp4/weights/best.pt')['model'].state_dict())

# set for inference
model.eval()    

image_files = glob.glob(os.path.join(image_folder, '*.*'))
# image_files = image_files[10:20]

results = model(image_files)
os.system("cls")
# results.show()

for pred, im in zip(results.xyxy, image_files):
    # no license plate was found
    if pred.numel() == 0:
        continue

    # found a plate
    pred = pred.tolist()
    image = cv2.imread(im)
    image_name = im.split("\\")[1]
    num = 0
    write_path = ""

    for box in pred:
        # box: [xmin, ymin, xmax, ymax, confidence, class number]
        bounding_box = box[:4]
        confidence = box[4]
        class_number = box[5]
        # print(f"Bounding Box Prediciton: {bounding_box}\tConfidence:{confidence:.2f}")

        xmin = ymin = xmax = ymax = 0
        # increase bounding box size if character detection
        if crop_folder == char_crop_folder:
            xmin = math.floor(bounding_box[0]) - 3
            ymin = math.floor(bounding_box[1]) - 3
            xmax = math.ceil(bounding_box[2]) + 3
            ymax = math.ceil(bounding_box[3]) + 3
            bounding_box = [xmin, ymin, xmax, ymax]
        else:
            xmin = math.floor(bounding_box[0])
            ymin = math.floor(bounding_box[1])
            xmax = math.ceil(bounding_box[2])
            ymax = math.ceil(bounding_box[3])

        cropped_image = crop_from_points(image, bounding_box) 
        # cv2.imshow("image", cropped_image)
        # cv2.waitKey(0)

        if crop_folder == lp_crop_folder:
            write_path = os.path.join(crop_folder, image_name) 
        else:
            write_path = f"{crop_folder}/character{num}-{image_name}"
        
        # cv2.imshow("Image", cropped_image)
        # cv2.waitKey(0)       
        cv2.imwrite(write_path, cropped_image)
        num += 1
results.save()

