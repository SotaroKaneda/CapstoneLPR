import os
import json
import scripts.utility as utils
from openpyxl import Workbook, load_workbook


data_path = r"C:\Users\Jed\Desktop\v2x-dataset\parsed_data.json"
annotation_folder = r"C:\Users\Jed\Desktop\v2x-dataset\cars_kp_annotations"
annotation_files = os.listdir(annotation_folder)
workbook_path = r"C:\Users\Jed\Desktop\v2x-dataset\ocr_comparison.xlsx"
annotations = []

for annotation_file in annotation_files:
    annotation_path = os.path.join(annotation_folder, annotation_file)
    annotation_data = utils.extract_from_datumaro(annotation_path)
    annotations += annotation_data
print(len(annotations))

# workbook = load_workbook(filename=workbook_path)
# sheet = workbook.active

with open(data_path, "r") as file:
    data_dict = json.load(file)
    keys = list(data_dict.keys())

    found = 0
    correct = 0
    incorrect = 0
    
    for annotation in annotations:
        image_name, labeled_plate, points = annotation

        if not image_name: continue

        for key in keys:
            associated_images = key.split(",")
            if image_name in associated_images:
                prediction = data_dict[key]["plate_num"]
                if prediction == labeled_plate:
                    correct += 1
                else:
                    incorrect += 1
                    pred_conf = data_dict[key]["read_conf"]
                    row = [labeled_plate, prediction, pred_conf, "", image_name]
                    row += [img for img in associated_images if img != image_name]
                    # sheet.append(row)
                    # print(labeled_plate, prediction, pred_conf, image_name, key)
                break

print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")

# workbook.save(filename=workbook_path)
