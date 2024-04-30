import torch
import os
import cv2
import sys
import glob
import re
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import scripts.utility as utils
from classifier.CharacterModel import CharacterModel
from pathlib import Path



def get_crop_lp(model_output, image):
    crop = np.array([])
    for prediction in model_output.xyxy:
        if prediction.numel() == 0:    
            continue

        prediction = prediction.tolist()     
        boxes = utils.get_bounding_box_data(prediction, image, padding=0, model="lp")

        # take highest conf box for license plates for now
        highest_conf = ""
        if len(boxes) > 1:
            highest_conf = boxes[0]
            for i in range(len(boxes) - 1):
                if boxes[i][1] < boxes[i + 1][1]:
                    highest_conf = boxes[i + 1]

            boxes = [highest_conf]

        for box in boxes:
            bbox, conf, klass = box
            crop = utils.crop_from_points(image, bbox)

    return crop

def get_crops_chars(model_output, image):
    char_list = []
    for prediction in model_output.xyxy:
        if prediction.numel() == 0:    
            continue

        prediction = prediction.tolist()    
        
        boxes = utils.get_bounding_box_data(prediction, image, padding=1, model="char")
        for box in boxes:
            bbox, conf, klass = box
            crop = utils.crop_from_points(image, bbox)
            char_list.append(crop)

    return char_list 


def predict_chars(character_crops, classifier, transforms, device):
    labels = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ" 
    pred_str = ""
    with torch.no_grad():
        for char_image in character_crops:
            char_image = image = transforms(image=char_image)["image"]
            char_image = char_image.unsqueeze(0).to(device)
            char_pred = classifier(char_image)
            values, real = torch.max(char_pred, 1)
            pred_str += labels[real]

    return pred_str

def pred_lp_trocr(lp_crop, trocr_model, processor, device):
    pixel_values = processor(images=lp_crop, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    generated_ids = trocr_model.generate(pixel_values)
    generated_ids = generated_ids.cpu()
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    parsed_ocr_value = re.sub('[\W_]+', '', generated_text[0])
    # lp_prediction = "".join(generated_text) 

    return parsed_ocr_value

def create_label_dict(file_lines):
    headers = file_lines[0].split(",")[:-1]
    data = file_lines[1:]
    label_dict = {}
    for line in data:
        UFM_ID,TXN_TIME,TOLLZONE_ID,LANE_POSITION,PLATE_TYPE,PLATE_TYPE_CONFIDENCE,PLATE_READ,PLATE_RDR_CONFIDENCE, \
        PLATE_JURISDICTION,IR_DISPOSITIONED,PAYMENT_METHOD,IMAGE1,IMAGE2,IMAGE3,IMAGE4,TYPE1,TYPE2,TYPE3,TYPE4 = line.split(",")[:-1]
        for image in [IMAGE1, IMAGE2, IMAGE3, IMAGE4]:
            if image != "None":
                label_dict[image.split(".")[0]] = PLATE_READ
    return label_dict

def HE(img):
    # Convert the image to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab)

        # Apply histogram equalization to the L channel
        l_equalized = cv2.equalizeHist(l)

        # Merge the equalized L channel with the original A and B channels
        lab_equalized = cv2.merge((l_equalized, a, b))

        # Convert the equalized LAB image back to BGR color space
        equalized_img = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)

        return equalized_img

def parse_csv(file_handle):
    lines = file_handle.readlines()
    headers = lines[0]
    data = lines[1:]
    num_records = len(data)

    return (headers, data, num_records)


# Path issue fix: https://stackoverflow.com/questions/57286486/i-cant-load-my-model-because-i-cant-put-a-posixpath
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# image_path = r"C:\Users\wilmc\Desktop\test-images\001115_1693566781560F5_031.jpg"
lp_weights = os.path.join("weights", "lp-detect.pt")
char_weights = os.path.join("weights", "char-detect.pt")
resnet_weights = os.path.join("weights", "resnet-classifier.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lp_model = torch.hub.load('ultralytics/yolov5', 'custom', lp_weights)#, force_reload=True)
char_model = torch.hub.load('ultralytics/yolov5', 'custom', char_weights)#, force_reload=True)

resnet_classifier = CharacterModel()
resnet_classifier.load_state_dict(torch.load(resnet_weights, map_location=torch.device('cpu')))
lp_model.eval()
char_model.eval()
resnet_classifier.eval()
lp_model.to(device)
char_model.to(device)
resnet_classifier.to(device)

image_size = 32
transforms = A.Compose([
        # A.Affine(shear={"x":15}, p=1.0),
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        ),
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,
        ),
        ToTensorV2()
    ])


# image_paths = glob.glob(os.path.join(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-LP\test\images", "*"))

# for image_path in image_paths:
for image_path in [os.path.join(r"D:\v2x-11-30-data\ALPRPlateExport11-30-23", "001015_1701090117276F05_941.jpg")]:
    base, image_name = os.path.split(image_path)
    name, ext = os.path.splitext(image_name)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    lp_pred = lp_model(image)
    lp_crop = get_crop_lp(lp_pred, image)
    
    if lp_crop.size > 0:
        inverted = ~lp_crop
        he = HE(lp_crop)
        inv_he = HE(inverted)
        char_pred = char_model(inv_he)
        char_crops = get_crops_chars(char_pred, he)
        pred = predict_chars(char_crops, classifier=resnet_classifier, transforms=transforms, device=device)
        print(pred)
            

pathlib.PosixPath = temp
        