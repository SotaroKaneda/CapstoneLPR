import os
import glob
import cv2
import sys
import matplotlib.pyplot as plt
import albumentations as A


def display_image(image, augmentations):
    maximum = len(augmentations)
    fig, ax = plt.subplots(1, maximum+1, figsize=(10,8))
    ax[0].imshow(image)
    for i in range(1, maximum+1):
        ax[i].imshow(augmentations[i-1])
    plt.tight_layout()
    plt.show()


image_folder = r"D:\v2x-11-30-data\small-test-set"
save_folder = r"D:\v2x-11-30-data\aug-small-test-set"
image_paths = glob.glob(os.path.join(image_folder, "*"))
image_size = 32
transforms = A.Compose([
    A.ColorJitter(p=0.3),
    A.RandomScale((0.7, 0.9)),
    A.Rotate(limit=(-15, 15), p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

for image_path in image_paths:
    image_name = image_path.split("\\")[-1]
    letter = image_name.split("-")[0]
    end = "-".join(image_name.split("-")[1:])
    # print(image_name)
    # augmentations = []
    image = cv2.imread(image_path)
    for i in range(10):
        img_t = transforms(image=image)["image"]
        save_path = os.path.join(save_folder, f"{letter}-aug{i}-{end}")
        # print(save_path)
        cv2.imwrite(save_path, img_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0]) 
        # augmentations.append(img_t)
    # display_image(image, augmentations)
