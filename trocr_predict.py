import os
import glob
import cv2
import time
import json
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


### Code from: https://huggingface.co/microsoft/trocr-large-printed

image_folder = r"C:\Users\Jed\Desktop\v2x-dataset\trocr_test_set"
save_path = r"C:\Users\Jed\Desktop\v2x-dataset" + r"\trocr_results.json"
# images = os.listdir(image_folder)
images = glob.glob(os.path.join(image_folder, "*"))
pred_dict = {}

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')

tic = time.time()
for img in images:
    image = cv2.imread(img) 
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # [:-4] slice is to remove the file extension
    key = img.split("\\")[-1][:-4]
    pred_dict[key] = generated_text

toc = time.time()

total_time = toc - tic
n_images = len(images)
average_time = total_time / n_images
print(f"Total Time: {total_time / 60} minutes\tAverage Time: {average_time} seconds for {n_images} images.")

with open(save_path, "w") as file:
    json.dump(pred_dict, file)
