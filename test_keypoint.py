import os
import cv2
import numpy as np
import time
from scripts.utility import extract_from_datumaro, visualize_annotations, keypoints_to_box, get_min_max


data_path = r"C:\Users\Jed\Desktop\kp_annotations.json"
data = extract_from_datumaro(data_path, 608)

image_folder = r"C:\Users\Jed\Desktop\v2x-dataset\cap-images"
image_names = []

for current_data in data[0:10]:
    image_name, plate_number, points = current_data

    if not points: continue

    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    top_left, top_right, bottom_left, bottom_right = points 
    input_points = np.float32([top_left, bottom_left, bottom_right, top_right])
    dest_points, width, height = keypoints_to_box(points)

    M = cv2.getPerspectiveTransform(input_points, dest_points)
    out = cv2.warpPerspective(image, M, (int(width), int(height)), flags=cv2.INTER_LINEAR)

    xmin, xmax, ymin, ymax = get_min_max(points)

    cropped = image[int(ymin):int(ymax), int(xmin):int(xmax)]

    cv2.namedWindow("Cropped", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Perspective Corrected", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cropped", 300, 100)
    cv2.resizeWindow("Perspective Corrected", 300, 100)

    visualize_annotations(image_path, keypoints=points)
    cv2.imshow("Cropped", cropped)
    cv2.imshow("Perspective Corrected", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



