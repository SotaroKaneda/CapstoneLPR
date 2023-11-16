import cv2
import glob
import os
import sys
import math


def crop_from_yolo_coords(image ,annotation_info):
    """
        image: image to be cropped. This is an array like object
        annotation_info: yolo format information
            format: space separated string "<class number> <box x center> <box y center> <box width> <box height>"
                box x center, box y center, box width, box height are normalized values between 0 and 1

        return value: cropped image -> array like
    """

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

    return image[ymin:ymax, xmin:xmax]

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
            
            cropped_image = crop_from_yolo_coords(image, info)
            cv2.imwrite(os.path.join(output_folder, f"{i}.{image_extension}"), cropped_image)


if __name__ == "__main__":
    main()


