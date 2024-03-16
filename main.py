import torch
import os
import cv2
import glob
import re
import time
import numpy as np
import scripts.utility as utils
import sys
import matplotlib
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def display_image(image):
    plt.imshow(image)
    plt.show()

def get_crops_lp(model_output, images, image_names):
    pred_dict = {}
    image_list = []
    license_plates = []
    for prediction, image, image_name in zip(model_output, images, image_names):
        image_dict = {}

        if prediction.numel() == 0:    
            # image_dict["lp_crop"] = []  
            continue

        prediction = prediction.tolist()     
        boxes = utils.get_bounding_box_data(prediction, image, padding=0)

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
            image_dict["lp_crop"] = crop
            license_plates.append(crop)

        pred_dict[image_name] = image_dict
        image_list.append(image_name)
    return pred_dict, image_list, license_plates


def get_crops_chars(model_output, license_plates, image_names, pred_dict, padding=0):
    for prediction, image, image_name in zip(model_output, license_plates, image_names):
        char_list = []
        if prediction.numel() == 0:    
            pred_dict[image_name]["char_crops"] = []  
            continue

        prediction = prediction.tolist()     
        boxes = utils.get_bounding_box_data(prediction, image, padding=padding)

        for box in boxes:
            bbox, conf, klass = box
            crop = utils.crop_from_points(image, bbox)
            char_list.append(crop)

        pred_dict[image_name]["char_crops"] = char_list  


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


class TestDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_RGB2BGR)
        image_name = self.image_paths[idx].split("\\")[-1].split(".")[0]
        return image, image_name

def custom_collate(batch):
    images, names = zip(*batch)
    return list(images), list(names)
    

image_paths = glob.glob(os.path.join(r"D:\v2x-11-30-data\ALPRPlateExport11-30-23", "*"))
label_path = os.path.join(r"C:\Users\Jed\Desktop\capstone_project\v2x-dataset", "data-11-30.csv")
with open(label_path, "r") as file:
    label_dict = create_label_dict(file.readlines())

pred_dict = ""
test_paths = image_paths[56396:]
test_data = TestDataset(image_paths)
batch_size = 16
test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=custom_collate)
batches = len(test_dataloader)

lp_weights = os.path.join("best_weights", "v-lp-detect-best.pt")
char_weights = os.path.join("best_weights", "x-char-detect-best-2.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tic = time.time()
lp_model = torch.hub.load('ultralytics/yolov5', 'custom', lp_weights)
char_model = torch.hub.load('ultralytics/yolov5', 'custom', char_weights)
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
lp_model.eval()
char_model.eval()
toc = time.time()
load_time = toc - tic

matplotlib.use("TkAgg")
tic = time.time()
batch_num = 0
original_ocr = []
with open("../trocr-3-10.csv", "w") as file:
    file.write("LABEL,OCR_PLATE,OCR_CHARACTERS,IMAGE\n")
    for batch in test_dataloader:
        if batch_num % 10 == 0:
            toc = time.time()
            print(f"Batch: {batch_num}/{batches}\tElapsed Time: {(toc - tic) / 60} minutes")
        images, image_names = batch
        # tic = time.time()
        lp_model.to(device)
        lp_results = lp_model(images)
        
        pred_dict, current_images, license_plates = get_crops_lp(lp_results.xyxy, images, image_names)

        char_model.to(device)
        char_results = char_model(license_plates)
        get_crops_chars(char_results.xyxy, license_plates, current_images, pred_dict, padding=1)

        trocr_model.to(device)
        for image_name, data in pred_dict.items():
            # ocr on lp images
            pixel_values = processor(images=data["lp_crop"], return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            generated_ids = trocr_model.generate(pixel_values)
            generated_ids = generated_ids.cpu()
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            parsed_ocr_value = re.sub('[\W_]+', '', generated_text)
            no_spaces = "".join(generated_text.split(" "))
            # original_ocr.append((generated_text, no_spaces, image_name))
            pred_dict[image_name]["ocr_plate"] = parsed_ocr_value

            # ocr on characters
            if data["char_crops"]:
                pixel_values = processor(images=data["char_crops"], return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(device)
                generated_ids = trocr_model.generate(pixel_values)
                generated_ids = generated_ids.cpu()
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
                character_prediction = "".join(generated_text)
                pred_dict[image_name]["ocr_chars"] = character_prediction
            else:
                pred_dict[image_name]["ocr_chars"] = ""

        for image_name, data in pred_dict.items():
            file.write(f'{label_dict[image_name]},{data["ocr_plate"]},{data["ocr_chars"]},{image_name}\n')

        batch_num += 1
toc = time.time()
total_time = toc - tic
print(f"Load Time: {load_time} seconds\tProcessing Time: {total_time} seconds for {batches} batches.")


# with open("../original_ocr.csv", "w") as file:
#     file.write("label,model_out,no_spaces,image")
#     for line in original_ocr:
#         file.write(f"{line[0]},{line[1]},{line[2]},{line[3]}\n")
