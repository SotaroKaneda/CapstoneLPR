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

def get_crops(model_output, images, padding=0, model_type=None):
    result_list = []
    char_list = []
    for prediction, image in zip(model_output, images):
        if prediction.numel() == 0:
            continue

        prediction = prediction.tolist()     
        boxes = utils.get_bounding_box_data(prediction, image, padding=padding)
        for box in boxes:
            bbox, conf, klass = box
            crop = utils.crop_from_points(image, bbox)

            if model_type == "char":
                char_list.append(crop)
            else:
                result_list.append(crop)

        if model_type == "char":
            result_list.append(char_list)
            char_list = []

    return result_list


class TestDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_RGB2BGR)


image_paths = glob.glob(os.path.join(r"D:\v2x-11-30-data\ALPRPlateExport11-30-23", "*"))

test_paths = image_paths[16:26]
test_data = TestDataset(test_paths)
test_dataloader = DataLoader(test_data, batch_size=16, collate_fn=lambda batch: batch)

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

for batch in test_dataloader:
    tic = time.time()
    lp_model.to(device)
    lp_results = lp_model(batch)
    license_plates = get_crops(lp_results.xyxy, batch, padding=0)

    char_model.to(device)
    char_results = char_model(license_plates)
    characters = get_crops(char_results.xyxy, license_plates, padding=1, model_type="char")

    # ocr on lp images
    pixel_values = processor(images=license_plates, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    trocr_model.to(device)
    generated_ids = trocr_model.generate(pixel_values)
    generated_ids = generated_ids.cpu()
    # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    parsed_ocr_values = [re.sub('[\W_]+', '', text) for text in generated_text]
    no_spaces = ["".join(value.split(" ")) for value in generated_text]
    print(no_spaces)
    print(parsed_ocr_values)


    # ocr on characters
    text_list = []
    for char_list in characters:
        pixel_values = processor(images=char_list, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        generated_ids = trocr_model.generate(pixel_values)
        generated_ids = generated_ids.cpu()
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        text_list.append(generated_text)

    character_predictions = ["".join(chars) for chars in text_list]
    toc = time.time()
    processing_time = toc - tic
    print(character_predictions)
    print()
    print(f"Load Time: {load_time} seconds\tProcessing Time: {processing_time} seconds.")
