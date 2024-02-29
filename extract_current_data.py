import os
import json
import openpyxl


images_path = r"C:\Users\Jed\Desktop\v2x-dataset\cap-images"
data_path = r"C:\Users\Jed\Desktop\v2x-dataset\data.csv"
write_path = r"C:\Users\Jed\Desktop\v2x-dataset\parsed_data.json"
write_path_excel = r""

image_names = os.listdir(images_path)
if "image_filenames.txt" in image_names:
    image_names.pop(image_names.index("image_filenames.txt"))
write_dict = {}

with open(data_path, "r") as f:
    headers = f.readline().split(",")
    headers[0] = "index"

    for line in f.readlines():
        line = line.split(",")
        INDEX, UFM_ID, TXN_TIME, TOLLZONE_ID, LANE_POSITION, PLATE_TYPE, PLATE_TYPE_CONFIDENCE, \
            PLATE_READ, PLATE_RDR_CONFIDENCE, PLATE_JURISDICTION, IR_DISPOSITIONED, PAYMENT_METHOD, IMAGE1, IMAGE2, \
            IMAGE3, IMAGE4, TYPE1, TYPE2, TYPE3, TYPE4 = line
        
        IMAGE1 = IMAGE1.split("/")[-1]
        IMAGE2 = IMAGE2.split("/")[-1]
        IMAGE3 = IMAGE3.split("/")[-1]
        IMAGE4 = IMAGE4.split("/")[-1]

        images = [IMAGE1, IMAGE2, IMAGE2, IMAGE2]

        for image_name in image_names:
            if image_name in images:
                write_dict[f"{IMAGE1},{IMAGE2},{IMAGE3},{IMAGE4}"] = {
                    "ufm_id": UFM_ID,
                    "txn_time": TXN_TIME,
                    "tollzone_id": TOLLZONE_ID,
                    "lane_position": LANE_POSITION,
                    "plate_num": PLATE_READ,
                    "read_conf": PLATE_RDR_CONFIDENCE,
                }
                break

with open(write_path, "w") as f:
    json.dump(write_dict, f)

