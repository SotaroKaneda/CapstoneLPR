import easyocr
import sys


def easyocr_test(image_path):
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image_path)
    for detection in result:
        if detection[2] > 0.5:
            # print(detection[1])
            return detection[1]

if __name__ == '__main__':
    image_path = sys.argv[1]
    easyocr_test(image_path)
