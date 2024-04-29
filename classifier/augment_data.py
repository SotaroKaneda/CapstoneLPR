import os
import glob
import cv2
import sys
import matplotlib.pyplot as plt
import albumentations as A


def display_image(image, augmentations):
    maximum = len(augmentations)
    fig, ax = plt.subplots(1, maximum, figsize=(10,8))
    ax[0].imshow(image)
    for i in range(1, maximum):
        ax[i].imshow(augmentations[i-1])
    plt.show()


image_names = []
root_folder = r"D:\v2x-11-30-data\char-dataset-new"
with open("wrong-prediction.csv", "r") as file:
    lines = file.readlines()
    headers = lines[0]
    data = lines[1:]

    for line in data:
        line = line.strip().split(",")
        label, prediction, image_name = line
        image_names.append(image_name)

characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_dist = {}
for c in characters:
    char_dist[c] = 0

image_size = 32
transforms = A.Compose([
    A.ColorJitter(p=0.5),
    A.Affine(shear=(0.3,0.4), p=0.5),
    A.RandomScale((0.7, 0.9)),
    A.Rotate(limit=(-15, 15), p=0.3),
    A.Sharpen(p=0.5),
    # A.LongestMaxSize(max_size=image_size),
    # A.PadIfNeeded(
    #     min_height=image_size,
    #     min_width=image_size,
    #     border_mode=cv2.BORDER_CONSTANT,
    #     value=(0, 0, 0),
    # ),
    # A.Normalize(
    #     mean=[0, 0, 0],
    #     std=[1, 1, 1],
    #     max_pixel_value=255,
    # ),
])

print(len(image_names))
for image_name in image_names:
    folder = image_name.split("-")[0]
    char_dist[folder] += 1
    name = "-".join(image_name.split("-")[1:])
    image_path = os.path.join(root_folder, folder, image_name)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    augmentations = []
    for i in range(1):
        augmented = transforms(image=image)["image"]
        img = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        save_path = os.path.join("aug", f"{folder}-aug{i}-{name}")
        cv2.imwrite(save_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0]) 
        # augmentations.append(augmented)
    # display_image(image, augmentations)
        
for key, value in char_dist.items():
    print(f"{key}:{value}")

