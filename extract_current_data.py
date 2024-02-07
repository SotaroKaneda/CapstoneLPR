import os
import json


images_path = r"C:\Users\Jed\Desktop\v2x-dataset\cap-images"
data_path = r"C:\Users\Jed\Desktop\v2x-dataset\data.csv"
write_path = r"C:\Users\Jed\Desktop\v2x-dataset\parsed_data.json"

images = os.listdir(images_path)
if "image_filenames.txt" in images:
    images.pop(images.index("image_filenames.txt"))
write_dict = {}

with open(data_path, "r") as f:
    headers = f.readline().split(",")
    headers[0] = "index"

    for line in f.readlines():
        INDEX, UFM_ID, TXN_TIME, TOLLZONE_ID, LANE_POSITION, PLATE_TYPE, PLATE_TYPE_CONFIDENCE, \
            PLATE_READ, PLATE_RDR_CONFIDENCE, IR_DISPOSITIONED, PAYMENT_METHOD, IMAGE1, IMAGE2, \
            IMAGE3, IMAGE4, TYPE1, TYPE2, TYPE3, TYPE4 = line.split(",")
        
        IMAGE1 = IMAGE1.split("/")[-1]
        IMAGE2 = IMAGE2.split("/")[-1]
        IMAGE3 = IMAGE3.split("/")[-1]
        IMAGE4 = IMAGE4.split("/")[-1]

        for img in images:
            if IMAGE1 == img or IMAGE2 == img or IMAGE3 == img or IMAGE4 == img:
                write_dict[f"{IMAGE1},{IMAGE2},{IMAGE3},{IMAGE4}"] = {
                    "plate_num": PLATE_READ,
                    "read_conf": PLATE_RDR_CONFIDENCE
                }
                break

with open(write_path, "w") as f:
    json.dump(write_dict, f)

