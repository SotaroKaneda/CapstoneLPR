import json
from openpyxl import Workbook, load_workbook
from scripts.utility import extract_from_datumaro


data_path = r"C:\Users\Jed\Desktop\v2x-dataset\parsed_data.json"
# annoation_path0 = r"C:\Users\Jed\Desktop\kp_annotations.json"
annotation_path1 = r"C:\Users\Jed\Desktop\v2x-dataset\kp-1300-1400.json"
annotation_path2 = r"C:\Users\Jed\Desktop\v2x-dataset\kp-1400-1500.json"
annotation_path3 = r"C:\Users\Jed\Desktop\v2x-dataset\kp-1500-1600.json"
workbook_path = r"C:\Users\Jed\Desktop\v2x-dataset\ocr_comparison.xlsx"
annotation_data1 = extract_from_datumaro(annotation_path1)
# annotation_data2 = extract_from_datumaro(annotation_path2)
# annotation_data3 = extract_from_datumaro(annotation_path3)
annotation_data = annotation_data1

# workbook = load_workbook(filename=workbook_path)
# sheet = workbook.active


# workbook = Workbook()
# sheet = workbook.active
# headers = ["Label", "OCR Prediction", "OCR Confidence", "Issue", "Label Image", "Associated Image 1", "Associated Image 2", "Associated Image 3"]
# sheet.title = "V2X Incorrect Dataset Entries"
# sheet.append(headers)

with open(data_path, "r") as file:
    data_dict = json.load(file)
    keys = list(data_dict.keys())

    found = 0
    correct = 0
    incorrect = 0
    
    for annotaion in annotation_data:
        image_name, labeled_plate, points = annotaion

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
                    print(labeled_plate, prediction, pred_conf, image_name, key)
                break

print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")

# workbook.save(filename=workbook_path)
