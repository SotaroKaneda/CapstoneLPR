import math


def crop_from_yolo_annotation(image ,annotation_info):
    """
        Uses data from a yolo format annotation file to return a cropped section of the input image.

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

        model_prediction: a YOLOv5 model prediction
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
        
        # floor, and ceil to slightly increase bounding box size
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