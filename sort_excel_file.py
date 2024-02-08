import openpyxl


workbook_path = r"C:\Users\Jed\Desktop\v2x-dataset\ocr_comparison.xlsx"
sorted_path = r"C:\Users\Jed\Desktop\v2x-dataset\ocr_comparison_sorted.xlsx"
workbook = openpyxl.load_workbook(filename=workbook_path)
sorted_workbook = openpyxl.Workbook()
sheet = workbook.active
sorted_sheet = sorted_workbook.active

headers = ["Label", "OCR Prediction", "OCR Confidence", "Issue", "Label Image", "Associated Image 1", "Associated Image 2", "Associated Image 3"]
sorted_sheet.title = "V2X Incorrect Dataset Entries"
sorted_sheet.append(headers)

group_incorrect = []
group_background = []
group_image_association = []
group_other = []

rows = list(sheet.values)[1:]
rows = [list(row) for row in rows]

for row in rows:
    label, pred, conf, issue, label_image, assoc1, assoc2, assoc3 = row
    
    if "Incorrect Prediction" in issue:
        group_incorrect.append(row)
    elif "Label background" in issue:
        group_background.append(row)
    elif "Wrong Image Association" in issue:
        group_image_association.append(row)
    else:
        group_other.append(row)

write_rows = group_incorrect + group_image_association + group_other + group_background

for row in write_rows:
    sorted_sheet.append(row)


sorted_workbook.save(filename=sorted_path)