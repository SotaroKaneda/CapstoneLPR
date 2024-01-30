import os
from scripts.utility import extract_from_datumaro, visualize_annotations


data_path = r"C:\Users\Jed\Desktop\kp_annotations.json"
data = extract_from_datumaro(data_path, 608)
image_folder = r"C:\Users\Jed\Desktop\v2x-dataset\cap-images"
image_name, plate_number, points = data[1]
image_path = os.path.join(image_folder, image_name)
print(image_path)

visualize_annotations(image_path, keypoints=points)


# image_file, plate_number, points = 
# print(image_file)
# print(plate_number)
# print(points)


