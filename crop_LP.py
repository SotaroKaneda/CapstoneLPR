import cv2
import glob
import os
import sys
import math


def yolo_to_coords(annotation_info):
    klass, x_center, y_center, box_width, box_height = annotation_info.split(' ')
    image_height, image_width, channels = image.shape
 
    x_center = float(x_center) * image_width
    y_center = float(y_center) * image_height
    box_width = float(box_width) * image_width
    box_height = float(box_height) * image_height

    xmin = math.ceil(x_center - (box_width/2))
    xmax = math.ceil(x_center + (box_width/2))
    ymin = math.ceil(y_center - (box_height/2))
    ymax = math.ceil(y_center + (box_height/2))

    return (xmin, xmax, ymin, ymax)

if len(sys.argv) != 4:
    print("Incorrect number of arguments.\nArguments: image_folder annotation_folder output_folder")
    sys.exit()

image_folder = sys.argv[1]
annotations_folder = sys.argv[2]
output_folder = sys.argv[3]

images = glob.glob(os.path.join(image_folder, "*.*"))
annotations = glob.glob(os.path.join(annotations_folder, "*.*"))

images.sort(key=lambda file : int(file.split('\\')[1].split('.')[0]))
annotations.sort(key=lambda file : int(file.split('\\')[1].split('.')[0]))


for i in range(len(images)):
    with open(annotations[i]) as f:
        image = cv2.imread(images[i])
        image_extension = images[i].split("\\")[1].split(".")[1]
        info = f.readline()
        xmin, xmax, ymin, ymax = yolo_to_coords(info)
        
        cropped_image = image[ymin:ymax, xmin:xmax]
        cv2.imwrite(os.path.join(output_folder, f"{i}.{image_extension}"), cropped_image)





