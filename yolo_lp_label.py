import torch
import glob
import os
import cv2
import math
import sys

"""
    Use YOLOv5 to create annotations for CVAT annotation tool
    arg1: YOLOv5 model weights file, example filename: best.pt
"""

if len(sys.argv) < 2:
    print("Error. Need to provide YOLOv5 model weights file.")
    sys.exit()

weights = sys.argv[1]

model = torch.hub.load('ultralytics/yolov5', 'custom', weights)

# load trained weights
# model.load_state_dict(torch.load('yolov5/runs/train/exp4/weights/best.pt')['model'].state_dict())

# set for inference
model.eval()    

image_files = glob.glob(os.path.join('lp_crops/crop200-300', '*.*'))
image_annotation_file = open("cvat-annotations/train.txt", "w")


batch1 = image_files[0:100]
batch2 = image_files[100:200]
batch3 = image_files[200:300]
batch4 = image_files[300:400]
batch5 = image_files[400:500]
batch6 = image_files[500:600]
batch7 = image_files[600:700]
batch8 = image_files[700:800]
batch9 = image_files[800:900]
batch10 = image_files[900:1000]
batch11 = image_files[1000:1100]
batch12 = image_files[1100:1200]
batch13 = image_files[1200:1300]
batch14 = image_files[1300:1400]
# batch15 = image_files[1400:1500]          # causes issues... some image wrong format or something????
batch16 = image_files[1500:1600]
batch17 = image_files[1600:1700]
batch18 = image_files[1700:1800]
batch19 = image_files[1800:1900]
batch20 = image_files[1900:2000]
batch21 = image_files[2000:2100]
batch22 = image_files[2100:2200]
batch23 = image_files[2200:2300]
batch24 = image_files[2300:2400]
batch25 = image_files[2400:2500]
batch26 = image_files[2500:2600]
batch27 = image_files[2600:2700]
batch28 = image_files[2700:2800]
batch29 = image_files[2800:2900]
batch30 = image_files[2900:3000]
batch31 = image_files[3000:3100]
batch32 = image_files[3100:3200]
batch33 = image_files[3200:3300]
batch34 = image_files[3300:3400]
batch35 = image_files[3400:3500]
batch36 = image_files[3500:3600]
batch37 = image_files[3600:3700]
batch38 = image_files[3700:3800]
batch39 = image_files[3800:3900]
batch40 = image_files[3900:4013]
current_batch = image_files


### Label in batches of 100 --> Hand Label --> retrain modle --> repeat
results = model(current_batch)
# results.show()

for pred, im in zip(results.xyxy, current_batch):
    # no object was found
    # continue... don't write an annotation file
    if pred.numel() == 0:
        continue
    
    # found an object
    pred = pred.tolist()     # convert tensor to a list... returns a 2d list with a 1d list per object found
                                # each object: [xmin, ymin, xmax, ymax, confidence, class number]

    # write class, x center, y center, box width, box height to corresponding annotation file
    ### CVAT Annotation folder format: https://opencv.github.io/cvat/docs/manual/advanced/formats/format-yolo/
    directory = "cvat-annotations/obj_train_data/"
    filename = im.split("\\")[-1]
    filename = filename.split(".")[0]   # remove image type ending

    # write image location for CVAT
    image_annotation_file.write(f"obj_train_data/{filename}.jpg\n")

    # write YOLO annotation files
    with open(f"{directory}/{filename}.txt", "w") as file:
        # for each bounding box in the model bounding box predictions
        for box in pred:
            bounding_box = box[:4]
            confidence = box[4]
            class_number = int(box[5])
            
            # top left point and bottom right point
            ### May change this... flooring top left point and rounding up bottom right point --> makes bounding box slightly bigger
            xmin = math.floor(bounding_box[0])
            ymin = math.floor(bounding_box[1])
            xmax = math.ceil(bounding_box[2])
            ymax = math.ceil(bounding_box[3])

            # load image and get image information
            image = cv2.imread(im)
            image_height, image_width, channels = image.shape

            #calculate center coordinate, box width, box height
            box_width = xmax - xmin
            box_height = ymax - ymin
            x_center = xmin + (box_width/2)
            y_center = ymin + (box_height/2)

            # normalize values
            box_width = box_width / image_width
            box_height = box_height / image_height
            x_center = x_center / image_width
            y_center = y_center / image_height
            
            # write yolo object annotation data to file
            # yolo format: class_number x_center y_center box_width box_height
            file.write(f"{class_number} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")


image_annotation_file.close()
# saves images with bounding boxes and confidence in /runs directory
#results.save()

