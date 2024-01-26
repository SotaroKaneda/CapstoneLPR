import cv2
import glob
import os
import sys
from utility import crop_from_yolo_annotation
    

"""
    This script takes an image folder, yolo annotations folder, and an ouput folder.
    Images are cropped using yolo annotation format and stored in the ouput folder.

    argument 1: image folder
    argument 2: annotations folder
    argument 3: output folder
"""

def main():
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
            
            cropped_image = crop_from_yolo_annotation(image, info)
            cv2.imwrite(os.path.join(output_folder, f"{i}.{image_extension}"), cropped_image)


if __name__ == "__main__":
    main()


