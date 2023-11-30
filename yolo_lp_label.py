import torch
import glob
import os
import cv2
import math

"""
    Use yolo to create annotations for CVAT annotation tool
"""

model = torch.hub.load('ultralytics/yolov5', 'custom', 'best7.pt')

# load trained weights
# model.load_state_dict(torch.load('yolov5/runs/train/exp4/weights/best.pt')['model'].state_dict())

# set for inference
model.eval()    

image_files = glob.glob(os.path.join('cap-images', '*.*'))
image_annotation_file = open("cvat-annotations/train.txt", "w")

batch1 = image_files[0:100]
batch2 = image_files[100:200]
batch3 = image_files[200:300]

current_batch = batch3

### Label in batches of 100 --> Hand Label --> retrain modle --> repeat
results = model(current_batch)
# results.show()

for pred, im in zip(results.xyxy, current_batch):
    # no license plate was found
    # continue... don't write an annotation file
    if pred.numel() == 0:
        continue
    
    # found a plate
    pred = pred.tolist()[0]     # convert tensor to a list... returns a 2d list with one element list
                                # pred: [xmin, ymin, xmax, ymax, confidence, class number]
    bounding_box = pred[:4]
    confidence = pred[4]
    class_number = int(pred[5])
    
    # top left point and bottom right point
    ### May change this... flooring top left point and rounding up bottom right point --> makes bounding box slightly bigger
    xmin = math.floor(bounding_box[0])
    ymin = math.floor(bounding_box[1])
    xmax = math.ceil(bounding_box[2])
    ymax = math.ceil(bounding_box[3])

    image = cv2.imread(im)

    image_height, image_width, channels = image.shape

    #calculate center coordinate, box width, box height
    box_width = xmax - xmin
    box_height = ymax - ymin
    x_center = xmin + (box_width/2)
    y_center = ymin + (box_height/2)

    # print(x_center, y_center, box_width, box_height)

    # rect_image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
    # center_dot = cv2.circle(rect_image, (int(x_center), int(y_center)), 5, (255, 0, 0))
    # cv2.imshow("Image", center_dot)
    # cv2.waitKey(0)

    # normalize values
    box_width = box_width / image_width
    box_height = box_height / image_height
    x_center = x_center / image_width
    y_center = y_center / image_height

    # print(x_center, y_center, box_width, box_height)
    # print(x_center*image_width, y_center*image_height, box_width*image_width, box_height*image_height)
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

    # write class, x center, y center, box width, box height to corresponding annotation file
    ### CVAT Annotation folder format: https://opencv.github.io/cvat/docs/manual/advanced/formats/format-yolo/

    directory = "cvat-annotations/obj_train_data/"
    filename = im.split("\\")[-1]
    filename = filename.split(".")[0]   # remove image type ending

    image_annotation_file.write(f"obj_train_data/{filename}.jpg\n")
    
    with open(f"{directory}/{filename}.txt", "w") as file:
        # yolo format: class_number x_center y_center box_width box_height
        file.write(f"{class_number} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")


image_annotation_file.close()
# saves images with bounding boxes and confidence in /runs directory
#results.save()

