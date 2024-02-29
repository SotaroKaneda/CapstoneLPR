import os
import shutil
import glob

"""
    Separate Images into folders of 100 images
"""

image_files = glob.glob(os.path.join(r"D:\v2x-11-30-data\ALPRPlateExport11-30-23", "*.*"))
num_images = len(image_files)
directory = r"D:\v2x-11-30-data\batches"
foldername = ""

# Create batch folders
dir_names = []
batches = []

for x in range(0, num_images, 100):
    images = image_files[x:x+100]

    if (x+100) > num_images:
        foldername = f"{directory}/new-batch{x}-{x + (num_images - x)}"
        batches.append(image_files[x:(x + (num_images - x))])
    else:
        foldername = f"{directory}/new-batch{x}-{x+100}"
        batches.append(image_files[x:x+100])

    dir_names.append(foldername)

    if not os.path.exists(foldername):
        os.mkdir(foldername)


for batch, folder in zip(batches, dir_names):
    for file in batch:
        print(file)
        filename = file.split("\\")[-1]
        shutil.copy2(file, os.path.join(folder, filename))