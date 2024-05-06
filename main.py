import torch
import os
import cv2
import sys
import glob
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import scripts.utility as utils
from classifier.CharacterModel import CharacterModel


# Path issue fix: https://stackoverflow.com/questions/57286486/i-cant-load-my-model-because-i-cant-put-a-posixpath
if sys.platform == "win32":
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
    lp_crop = utils.get_crop_lp(lp_pred, image)
    
    if lp_crop.size > 0:
        inverted = ~lp_crop
        he = utils.HE(lp_crop)
        inv_he = utils.HE(inverted)
        char_pred = char_model(inv_he)
        char_crops = utils.get_crops_chars(char_pred, he)
        pred = utils.predict_chars(char_crops, classifier=resnet_classifier, transforms=transforms, device=device)
        print(pred)
            

if sys.platform == "win32":
    pathlib.PosixPath = temp
        