# YOLO Setup: https://medium.com/analytics-vidhya/training-a-custom-object-detection-model-with-yolo-v5-aa9974c07088
# YOLO Custom Training: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#local-logging
# YOLO Object Detection Tutorial: https://docs.ultralytics.com/datasets/detect/
# Inference using custom trained model: https://github.com/ultralytics/yolov5/issues/7044


import torch
import glob
import os
import cv2
import numpy as np
from easyocr_test import easyocr_test

# fix on silo... issues with pillow library
if os.name == "posix":
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

model = torch.hub.load('ultralytics/yolov5', 'custom', 'best5.pt')

# load trained weights
# model.load_state_dict(torch.load('yolov5/runs/train/exp4/weights/best.pt')['model'].state_dict())

# set for inference
model.eval()    

image_files = glob.glob(os.path.join('/l/research/v2x/images', '*.*'))
batch1 = image_files[0:250]

results = model(batch1)
# results.show()

# for pred, im in zip(results.xyxy, batch1):
#     pred = pred.tolist()[0]
#     # pred: [xmin, ymin, xmax, ymax, confidence, class number]
#     bounding_box = pred[:4]
#     confidence = pred[4]
#     class_number = pred[5]
#     print(f"Bounding Box Prediciton: {bounding_box}\tConfidence:{confidence:.2f}")
    
#     # display cropped image
#     xmin = int(bounding_box[0])
#     ymin = int(bounding_box[1])
#     xmax = int(bounding_box[2])
#     ymax = int(bounding_box[3])

#     image = cv2.imread(im)
#     cropped = image[ymin:ymax, xmin:xmax]
#     # both = np.concatenate((image, cropped), axis=0)
#     cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow("Image", 300, 300)
#     cv2.imshow("Image", cropped)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


#help(results)

# results.save()

