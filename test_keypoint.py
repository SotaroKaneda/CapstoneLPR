from scripts.utility import extract_from_datumaro


data_path = r"C:\Users\Jed\Desktop\kp_annotations.json"
data = extract_from_datumaro(data_path, 608)

print(data[0])

# image_file, plate_number, points = 
# print(image_file)
# print(plate_number)
# print(points)


