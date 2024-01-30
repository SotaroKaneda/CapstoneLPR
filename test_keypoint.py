import os
import cv2
import numpy as np
from scripts.utility import extract_from_datumaro, visualize_annotations, keypoints_to_box


data_path = r"C:\Users\Jed\Desktop\kp_annotations.json"
data = extract_from_datumaro(data_path, 5)
print(data)
image_folder = r"C:\Users\Jed\Desktop\v2x-dataset\cap-images"
image_name, plate_number, points = data[4]
image_path = os.path.join(image_folder, image_name)
image = cv2.imread(image_path)

top_left, top_right, bottom_left, bottom_right = points 
input_points = np.float32([top_left, bottom_left, bottom_right, top_right])
dest_points, width, height = keypoints_to_box(points)

M = cv2.getPerspectiveTransform(input_points, dest_points)
out = cv2.warpPerspective(image, M, (int(width), int(height)), flags=cv2.INTER_LINEAR)

cv2.imshow("Perspective Corrected", out)
cv2.waitKey(0)
cv2.destroyAllWindows()



# visualize_annotations(image_path, keypoints=points)


