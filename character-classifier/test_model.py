import os
import glob
import cv2
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader
from CharacterDataset import CharacterDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
from CharacterModel import CharacterModel


def main():
    image_paths = []
    root = r"D:\v2x-11-30-data\char-dataset-new"
    folders = os.listdir(root)
    for folder in folders:
        image_paths += glob.glob(os.path.join(root, folder, "*"))

    image_size = 32
    transforms = A.Compose([
        # A.Affine(shear={"x":15}, p=1.0),
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        ),
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,
        ),
        ToTensorV2()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 100
    num_workers = 8
    model = CharacterModel()
    model.load_state_dict(torch.load(r"C:\Users\Jed\Desktop\capstone_project\character-classifier\resnet50-100.pth"))
    model.eval()
    model.to(device)

    dataset = CharacterDataset(image_paths, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)#, shuffle=True)

    correct = 0
    incorrect = 0
    num_batches = len(dataloader)
    num_characters = len(image_paths)
    wrong = []

    print(f"Number of Images: {num_characters}\tNumber of Batches: {num_batches}")

    for index, batch in enumerate(dataloader):
        images, labels, image_names = batch
        with torch.no_grad():
            # image = images[0].permute(1, 2, 0).numpy()  # Permute dimensions for imshow (C, W, H) to (W, H, C)
            images = images.to(device)
            # label = dataset.labels[labels.item()]
            outputs = model(images)
            values, real = torch.max(outputs, 1)

            for label, pred, image_name in zip(labels, real, image_names):
                if label == pred:
                    correct += 1
                else:
                    incorrect += 1
                    wrong.append(f"{dataset.labels[label]},{dataset.labels[pred]},{image_name}\n")

            print(f"Batch {index} / {num_batches} finished.")

            # plt.imshow(image)
            # plt.title(f"Prediction: {prediction}    Label: {label}")
            # plt.show()

    with open("wrong-prediction.csv", "w") as file:
        file.write(f"LABEL,PREDICTION,IMAGE\n")
        for line in wrong:
            file.write(line)

    print()
    print(f"Correct: {correct}/{num_characters}\tIncorrect: {incorrect}/{num_characters}")
    print(f"Precentage Correct: {(correct / num_characters) * 100:0.5f}")

if __name__ == "__main__":
    main()