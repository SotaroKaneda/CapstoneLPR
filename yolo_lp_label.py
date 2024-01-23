import torch
import glob
import os
import cv2
import math
import sys
# from utility import get_bounding_box_data

"""
    Use YOLOv5 to create annotations for CVAT annotation tool
    arg1: YOLOv5 model weights file, example filename: best.pt
"""

if len(sys.argv) < 2:
    print("Error. Need to provide YOLOv5 model weights file.")
    sys.exit()

weights = sys.argv[1]

model = torch.hub.load('ultralytics/yolov5', 'custom', weights)


# set for inference
model.eval()    

image_files = glob.glob(os.path.join('../v2x-dataset/cap-images', '*.*'))
image_annotation_file = open("cvat-annotations/train.txt", "w")

batches = [image_files[i:i+100] for i in range(len(image_files))]
print(len(batches))
print(batches[:10])
print(batches[-10:])

# results = model(current_batch)

# for pred, im in zip(results.xyxy, current_batch):
#     # no object was found
#     # continue... don't write an annotation file
#     if pred.numel() == 0:
#         continue
    
#     # found an object
#     pred = pred.tolist()  

#     # extract bounding box data from the prediction
#     boxes = get_bounding_box_data(pred, 0)

#     # write class, x center, y center, box width, box height to corresponding annotation file
#     ### CVAT Annotation folder format: https://opencv.github.io/cvat/docs/manual/advanced/formats/format-yolo/
#     directory = "cvat-annotations/obj_train_data/"
#     filename = im.split("\\")[-1]
#     filename = filename.split(".")[0]   # remove image type ending

#     # write image location for CVAT
#     image_annotation_file.write(f"obj_train_data/{filename}.jpg\n")

#     # write YOLO annotation files
#     with open(f"{directory}/{filename}.txt", "w") as file:
#         # for each bounding box in the model bounding box predictions
#         for box in boxes:
#             bounding_box = box[0]
#             confidence = box[1]
#             class_number = int(box[2])
            
#             xmin, ymin, xmax, ymax = bounding_box

#             # load image and get image information
#             image = cv2.imread(im)
#             image_height, image_width, channels = image.shape

#             #calculate center coordinate, box width, box height
#             box_width = xmax - xmin
#             box_height = ymax - ymin
#             x_center = xmin + (box_width/2)
#             y_center = ymin + (box_height/2)

#             # normalize values
#             box_width = box_width / image_width
#             box_height = box_height / image_height
#             x_center = x_center / image_width
#             y_center = y_center / image_height
            
#             # write yolo object annotation data to file
#             # yolo format: class_number x_center y_center box_width box_height
#             file.write(f"{class_number} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")


# image_annotation_file.close()


