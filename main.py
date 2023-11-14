# YOLO Setup: https://medium.com/analytics-vidhya/training-a-custom-object-detection-model-with-yolo-v5-aa9974c07088
# YOLO Custom Training: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#local-logging
# YOLO Object Detection Tutorial: https://docs.ultralytics.com/datasets/detect/
# Inference using custom trained model: https://github.com/ultralytics/yolov5/issues/7044


import torch
import glob
import os

model = torch.hub.load('ultralytics/yolov5', 'custom', 'best1.pt')

# load trained weights
# model.load_state_dict(torch.load('yolov5/runs/train/exp4/weights/best.pt')['model'].state_dict())

# set for inference
model.eval()    

image_files = glob.glob(os.path.join('car_images', '*.*'))

results = model(image_files)
print(results.xyxy[0])
# results.save()

