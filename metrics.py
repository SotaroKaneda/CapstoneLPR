import torch
import os
import glob
import cv2
from torchvision import ops
from scripts.utility import convert_yolo_annotation_data_to_points
import numpy as np
import math
import sys


"""
    argument 1: images folder
    argument 2: truth folder
    argument 3: prediction folder
    argument 4: threshold
"""


def get_iou(prediction_box, ground_truth_box):
    """
        ** TODO: ADD DOCSTRING**
    """
    truth_box_tensor = torch.tensor([ground_truth_box], dtype=torch.float)
    prediction_box_tensor = torch.tensor([prediction_box], dtype=torch.float)

    iou = ops.box_iou(truth_box_tensor, prediction_box_tensor)

    return iou.numpy()[0][0].item()


image_folder = sys.argv[1]
truth_folder = sys.argv[2]
prediciton_folder = sys.argv[3]

images = glob.glob(os.path.join(image_folder, '*.*'))
truth_files = glob.glob(os.path.join(truth_folder, '*.*'))
prediction_files = glob.glob(os.path.join(prediciton_folder, '*.*'))

TP = 0      # IOU > threshold
FP = 0      # IOU < threshold
TN = 0      # Model predicted no object, actually no object, no license plates
FN = 0      # Model predicted no object, but there actually is an object
multiple_boxes = 0
iou_list = []

for image in images:
    # directory = image.split("\\")[0]
    image_name = image.split("\\")[-1].split(".")[0]
    label_file_name = f"{image_name}.txt"
    image_extension = image.split("\\")[-1].split(".")[1]
    img = cv2.imread(image)

    # cvat keeps files even if they don't have an object in them... so empty file
    # if label_file_name in truth_files:
    #     ...

    truth_file = os.path.join(truth_folder, label_file_name)
    prediction_file = os.path.join(prediciton_folder, label_file_name)

    # model didn't find an object in the image
    if prediction_file not in prediction_files:
        # check truth file
        tf = open(truth_file)
        if tf.readline() == "":
            TN += 1
        else:
            FN += 1

        tf.close()
    else:
        # prediction found
        with open(prediction_file) as pf, open(truth_file) as tf:
            truth_info = tf.readline()
            prediction_info = pf.readlines()

            # the model has predicted more than one bounding box
            if len(prediction_info) > 1:
                multiple_boxes += 1

                # check boxes to see if one lines up
                for line in prediction_info:
                    truth_box = convert_yolo_annotation_data_to_points(img, truth_info)
                    prediction_box = convert_yolo_annotation_data_to_points(img, line)
                    
                    iou = get_iou(prediction_box, truth_box)
                    iou = get_iou(prediction_box, truth_box)
                    iou_list.append(iou)
            else:
                prediction_info = prediction_info[0]

                # no object in the image
                if truth_info == "":
                    continue

                truth_box = convert_yolo_annotation_data_to_points(img, truth_info)
                prediction_box = convert_yolo_annotation_data_to_points(img, prediction_info)

                iou = get_iou(prediction_box, truth_box)
                iou_list.append(iou)



# Calculate metrics for different threshold values
print("Counting Multiple Bounding Boxes as FP")
print(f"Number of Multiple Boxes: {multiple_boxes}")
print("Threshold\tRecall\tPrecision\tTP\tFP\tTN\tFN")
print()

p_TN = TN
p_FN = FN
threshold_list = np.arange(0.05, 1.0, 0.05)
threshold_list = np.append(threshold_list, [0.92])
# for threshold in sorted(threshold_list, reverse=True):
for threshold in [0.92]:

    for iou in iou_list:
        if iou > threshold:
            TP += 1
        else:
            FP += 1

    # precision = correct prediction / total predictions = TP/(TP + FP)
    precision = TP / (TP + FP)
    # Recall = correct predictions / total ground truth = TP / (TP + FN)
    recall = TP / (TP + FN)
    
    print(f"{threshold:.2f}\t\t{recall:.2f}\t{precision:.2f}\t\t{TP}\t{FP}\t{TN}\t{FN}")
    TP = 0
    FP = 0
    TN = p_TN
    FN = p_FN



