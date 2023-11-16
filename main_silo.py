# YOLO Setup: https://medium.com/analytics-vidhya/training-a-custom-object-detection-model-with-yolo-v5-aa9974c07088
# YOLO Custom Training: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#local-logging
# YOLO Object Detection Tutorial: https://docs.ultralytics.com/datasets/detect/
# Inference using custom trained model: https://github.com/ultralytics/yolov5/issues/7044


import torch
import glob
import os
import cv2
import numpy as np
import pandas as pd
import csv
import time
from easyocr_test import easyocr_test
from crop_LP import crop_from_points

# fix on silo... issues with pillow library
# https://forums.fast.ai/t/oserror-image-file-is-truncated-38-bytes-not-processed/30806/5
if os.name == "posix":
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

model = torch.hub.load('ultralytics/yolov5', 'custom', 'best5.pt')

s0_df = pd.read_csv("../../s0_rel.csv")
s1_df = pd.read_csv("../../s1_rel.csv")
actual = ""
write_data = []
no_LP = []
true = 0
false = 0
no_plate = 0
fields = ["Image", "Actual", "Prediciton", "True"]

# load trained weights
# model.load_state_dict(torch.load('yolov5/runs/train/exp4/weights/best.pt')['model'].state_dict())

# set for inference
model.eval()    

image_files = glob.glob(os.path.join('/l/research/v2x/images', '*.*'))
batch1 = image_files[0:250]
batch2 = image_files[250:500]
batch3 = image_files[500:750]
batch4 = image_files[750:1000]
batch5 = image_files[1000:1250]
batch6 = image_files[1250:1500]
batch7 = image_files[1500:1750]
batch8 = image_files[1750:2000]
batch9 = image_files[2000:2250]
batch10 = image_files[2250:2500]
batch11 = image_files[2500:2750]
batch12 = image_files[2750:3000]
batch13 = image_files[3000:3250]
batch14 = image_files[3250:3500]
batch15 = image_files[3500:3750]
batch16 = image_files[3750:4000]
batch17 = image_files[4000:]

batches = [batch1, batch2, batch3, batch4, batch5, batch6, batch7,
           batch8, batch9, batch10, batch11, batch12, batch13, batch14,
           batch15, batch16, batch17]


for batch in batches:
    results = model(batch)
    # results.show()

    for pred, im in zip(results.xyxy, batch):
        if pred.tolist():
            image_name = im.split("/")[-1]
            pred = pred.tolist()[0]
            # pred: [xmin, ymin, xmax, ymax, confidence, class number]
            bounding_box = pred[:4]
            confidence = pred[4]
            class_number = pred[5]
            #print(f"Bounding Box Prediciton: {bounding_box}\tConfidence:{confidence:.2f}")

            image = cv2.imread(im)
            cropped = crop_from_points(image, bounding_box)
            cv2.imwrite(f"../../cropped_images/{image_name}.jpg", cropped)

            # look for current image in image files
            look1 = s0_df.loc[s0_df["IMAGE1"] == image_name]
            look2 = s0_df.loc[s0_df["IMAGE2"] == image_name]
            look3 = s1_df.loc[s1_df["IMAGE1"] == image_name]
            look4 = s1_df.loc[s1_df["IMAGE2"] == image_name]

            if not look1.empty: 
                found = s0_df.loc[s0_df["IMAGE1"] == image_name]["PLATE_READ"].to_list()[0] 
            elif not look2.empty: 
                found = s0_df.loc[s0_df["IMAGE2"] == image_name]["PLATE_READ"].to_list()[0]
            elif not look3.empty:
                found = s1_df.loc[s1_df["IMAGE1"] == image_name]["PLATE_READ"].to_list()[0]
            elif not look4.empty: 
                found = s1_df.loc[s1_df["IMAGE2"] == image_name]["PLATE_READ"].to_list()[0]

            # easyocr prediction
            prediction = easyocr_test(f"../../cropped_images/{image_name}.jpg")
            if prediction != None:
                prediction = prediction.replace(" ", "")

            print(prediction, found)
            # calculate correct and incorrect predictions
            if found == prediction: true += 1
            else: false += 1

            write_data.append([image_name, found, prediction, found == prediction])
        else:
            no_plate += 1
            no_LP.append([image_name])

        # # both = np.concatenate((image, cropped), axis=0)
        # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        # # cv2.resizeWindow("Image", 300, 300)
        # cv2.imshow("Image", cropped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

with open("preds.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)  
    csvwriter.writerow(fields)
    csvwriter.writerows(write_data) 

with open("no_LP.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)  
    csvwriter.writerow(["image"])
    csvwriter.writerows(no_LP) 

print(len(write_data))
print(len(no_LP))
print(f"True: {true}\tFalse:{false}\tNo Plates: {no_plate}")
#help(results)

# results.save()

