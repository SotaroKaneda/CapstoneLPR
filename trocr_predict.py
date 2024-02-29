import os
import sys
import glob
import cv2
import time
import re
import torch
# import json
import openpyxl
import scripts.utility as utils
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


### Code from: https://huggingface.co/microsoft/trocr-large-printed

if len(sys.argv) < 3:
    print("Error. Arguments: image_folder_path, save_file_path")
    sys.exit()

image_folder = sys.argv[1]
save_file_path = sys.argv[2]
save_file_name = "trocr-new-data.xlsx"
save_path = os.path.join(save_file_path, save_file_name)
images = glob.glob(os.path.join(image_folder, "*"))
pred_dict = {}
workbook = ""
sheet = ""
headers = ["Label", "OCR", "Image"]

# if not os.path.isfile(save_path):
#     workbook = utils.create_init_workbook("TrOCR and Labels", headers=headers)
# else:
#     workbook = openpyxl.load_workbook(save_path)

# sheet = workbook.active

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')

model.to(device)

tic = time.time()
iterations = 0
for img in images:
    image = cv2.imread(img) 
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    generated_ids = model.generate(pixel_values)
    generated_ids = generated_ids.cpu()

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    parsed_ocr_value = re.sub('[\W_]+', '', generated_text)
    
    # [:-4] slice is to remove the file extension
    key = img.split("\\")[-1][:-4]
    row = ["", parsed_ocr_value, key]
    # sheet.append(row)

    # pred_dict[key] = generated_text

    iterations += 1

    if (iterations % 100) == 0:
        toc = time.time()
        print(f"Time: {toc - tic} seconds")
        print(f"Iterations: {iterations}")
        # workbook.save(filename=save_path)

toc = time.time()

total_time = toc - tic
n_images = len(images)
average_time = total_time / n_images
# print(f"Total Time: {total_time / 60} minutes\tAverage Time: {average_time} seconds for {n_images} images.")


# workbook.save(filename=save_path)
# with open(save_path, "w") as file:
#     json.dump(pred_dict, file)
