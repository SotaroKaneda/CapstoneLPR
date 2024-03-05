# YOLO Setup: https://medium.com/analytics-vidhya/training-a-custom-object-detection-model-with-yolo-v5-aa9974c07088
# YOLO Custom Training: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#local-logging
# YOLO Object Detection Tutorial: https://docs.ultralytics.com/datasets/detect/
# Inference using custom trained model: https://github.com/ultralytics/yolov5/issues/7044

import torch
import glob
import os
import sys
import cv2
import matplotlib
import matplotlib.pyplot as plt
# from easyocr_test import easyocr_test
from scripts.utility import get_bounding_box_data, crop_from_points


"""
    This script accepts either a single image or a folder of images to run YOLOv5 object detection on.
    argument 1: an image file name or image folder name
    argument 2: whether to run license plate(lp) detection or character detection(char)
        options: lp, char
"""

if len(sys.argv) < 3:
    print("Error. Supply image or image folder and a detection mode(lp for license plate, char for character)")
    sys.exit()

image_files = ""
weights = ""
save_folder = r"D:\v2x-11-30-data\characters"

# is the input a file or directory?
if os.path.isdir(sys.argv[1]):
    image_files = glob.glob(os.path.join(sys.argv[1], '*.*'))
else:
    image_files = [sys.argv[1]]

# multiples = []
# with open("multiple_images.txt", "r") as file:
#     for img in file.readlines():
#         multiples.append(img.strip())

# current_crops = os.listdir(save_folder)
# current = current_crops + multiples

# print(f"num total: {len(image_files)}")
# print(f"num current: {len(current)}")
# new_images = []
# for img in image_files:
#     if img.split("\\")[-1].split(".")[0] + ".png" not in current:
#         new_images.append(img)

# print(f"new: {len(new_images)}")


# load weights
if sys.argv[2] == "lp":
    weights = os.path.join("best_weights", "v-lp-detect-best.pt")
elif sys.argv[2] == "char":
    weights = os.path.join("best_weights", "x-char-detect-best-2.pt")
else:
    print("Error. Incorrect detection mode specified.")
    sys.exit()


model = torch.hub.load('ultralytics/yolov5', 'custom', weights)

# matplotlib.use('TkAgg')
model.eval()    

batch_size = 200
num_images = len(image_files)
batches = []
no_plate = 0

for i in range(0, num_images, batch_size):
    if (i + batch_size) > num_images:
        batches.append(image_files[i:])
    else:
        batches.append(image_files[i:(i+batch_size)])

batch_num = 0

batches = [image_files[:10]]
# with open("multiple_images.txt", "w") as file, open("no_plate.txt", "w") as n_file:
for batch in batches:
    batch_num += 1

    if (batch_num % 100) == 0:
        print(batch_num)

    results = model(batch)

    for prediction, image in zip(results.xyxy, batch):
        image_name = image.split("\\")[-1].split(".")[0] + ".png"
        
        if prediction.numel() == 0:
            no_plate += 1
            # n_file.write(f"{image_name}\n")
            continue

        prediction = prediction.tolist()     
        boxes = get_bounding_box_data(prediction, padding=1)

        img = cv2.imread(image)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_name_base = image.split("\\")[-1].split(".")[0]
        image_name = image.split("\\")[-1].split(".")[0] + ".png"
        os.mkdir(os.path.join(save_folder, image_name_base))
        # save_path = os.path.join(save_folder, image_name)

        # if len(boxes) > 1:
            # file.write(f"{image_name}\n")
        #     continue

        for index, box in enumerate(boxes):
            bbox, conf, klass = box
            crop = crop_from_points(img, bbox)
            save_path = os.path.join(save_folder, image_name_base, f"{index}-" + image_name)
            cv2.imwrite(save_path, crop, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
