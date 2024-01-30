import math
import json
import cv2
import numpy as np


# TODO Add image boundary checking in annotation_to_points and get_bounding_box_data
# TODO Add vertical character sorting to get_bounding_box_data

def annotation_to_points(image, annotation_info):
    
    klass, x_center, y_center, box_width, box_height = annotation_info.split(' ')
    image_height, image_width, channels = image.shape
 
    x_center = float(x_center) * image_width
    y_center = float(y_center) * image_height
    box_width = float(box_width) * image_width
    box_height = float(box_height) * image_height

    xmin = math.ceil(x_center - (box_width/2))
    ymin = math.ceil(y_center - (box_height/2))
    xmax = math.ceil(x_center + (box_width/2))
    ymax = math.ceil(y_center + (box_height/2))

    return [xmin, ymin, xmax, ymax]


def crop_from_yolo_annotation(image, annotation_info):
    """
        Uses data from a yolo format annotation file to return a cropped section of the input image.

        image: image to be cropped. This is an array like object
        annotation_info: yolo format information
            format: space separated string "<class number> <box x center> <box y center> <box width> <box height>"
                box x center, box y center, box width, box height are normalized values between 0 and 1

        return value: cropped image -> array like
    """

    xmin, ymin, xmax, ymax = annotation_to_points(image, annotation_info)

    return image[ymin:ymax, xmin:xmax]


def crop_from_points(image, bbox_points):
    """
        Uses four bounding box points to return a cropped section from the input image.

        image: image to crop -> array like
        bbox_points: [xmin, ymin, xmax, ymax] -> list

        return: a cropped image -> array like
    """
    
    xmin, ymin, xmax, ymax = bbox_points

    return image[int(ymin):int(ymax), int(xmin):int(xmax)]


def get_bounding_box_data(model_prediction, padding=0):
    """
        Retrieves bounding box data from a YOLOv5 model prediction output.
        Optionally adds padding to the bounding boxes

        model_prediction: a YOLOv5 model prediction. Format: [[xmin, ymin, xmax, ymax, confidence, class number]]
        padding: optional parameter to add padding to the bounding box. This increases the size of the bounding box.

        return: boxes list with bounding box list, confidence, and class number per box
                [[bounding_box, confidence, class_number], ...]
                [[xmin, ymin, xmax, ymax], confidence, class_number], ...]
                len(boxes) >= 1
    """
    
    boxes = []

    # for each bounding box predicted in the image
    for box in model_prediction:
        # box: [xmin, ymin, xmax, ymax, confidence, class number]
        bounding_box = box[:4]
        confidence = box[4]
        class_number = box[5]       # only two for now: license plate or character
        
        xmin = math.floor(bounding_box[0]) - padding
        ymin = math.floor(bounding_box[1]) - padding
        xmax = math.ceil(bounding_box[2]) + padding
        ymax = math.ceil(bounding_box[3]) + padding

        bounding_box = [xmin, ymin, xmax, ymax, confidence, class_number]
        boxes.append(bounding_box)

    # TODO: Add vertical sorting. This does not account for vertically oriented boxes that appear on some license plates
    # sort the boxes by the xmin corrdinate to order the boxes correctly left to right
    if len(boxes) > 1:
        boxes.sort(key=lambda box: box[0])

    return boxes


def extract_from_datumaro(json_file, finished_items=None):
    f = open(json_file)
    json_dict = json.load(f)
    
    data = []
    items = json_dict["items"]

    if finished_items:
        items = items[:finished_items]

    for item in items:
        id = item["id"]
        image_file = f"{id.split('/')[-1]}.jpg"
        annotations = item["annotations"]
        plate_number = ""
        points = []
        
        # check for labeled images
        if annotations:
            plate_number = annotations[0]["attributes"]["plate number"]
            pts = annotations[0]["points"]
            
            for i in range(0, 8, 2):
                points.append([pts[i], pts[i + 1]])

            ### sort points: [top left, top right, bottom left, bottom right]
            # sort by y coordinate
            points.sort(key=lambda point: point[1])
            
            # sort each half by x corrdinate
            top = points[:2]
            top.sort(key=lambda point: point[0])
            bottom = points[2:]
            bottom.sort(key=lambda point: point[0])

            points = top + bottom
            

        data.append([image_file, plate_number, points])
    
    return data


def keypoints_to_box(keypoints, padding=None):
    top_left, top_right, bottom_left, bottom_right = keypoints
    xmin = min(top_left[0], bottom_left[0])
    xmax = max(top_right[0], bottom_right[0])
    ymin = min(top_left[1], top_right[1])
    ymax = max(bottom_left[1], bottom_right[1])
    box_width = xmax - xmin
    box_height = ymax - ymin
    
    # point order: top left, bottom left, bottom right, top right 
    dest_points = np.float32([[0, 0],
                              [0, box_height-1],
                              [box_width-1, box_height-1],
                              [box_width-1, 0]])

    return [dest_points, box_width, box_height]


def visualize_annotations(image_path, box=None, keypoints=None, color=(0, 0, 255)):
    image = cv2.imread(image_path)
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    
    if keypoints:
        print(keypoints)
        image = cv2.rectangle(image, tuple(keypoints[0]), tuple(keypoints[3]), (0, 255, 0), 3)

        for point in keypoints:
            x, y = point
            image = cv2.circle(image, (x, y), radius=10, color=color, thickness=-1)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()