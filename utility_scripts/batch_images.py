import os
import shutil
import glob

"""
    Separate Images into folders of 100 images
"""

image_files = glob.glob(os.path.join('cap-images', '*.*'))
num_images = len(image_files)
directory = "cap-image-batched"
foldername = ""

# Create batch folders
dir_names = []
batches = []

for x in range(0, num_images, 100):
    images = image_files[x:x+100]

    if (x+100) > num_images:
        foldername = f"{directory}/batch{x}-{x + (num_images - x)}"
        batches.append(image_files[x:(x + (num_images - x))])
    else:
        foldername = f"{directory}/batch{x}-{x+100}"
        batches.append(image_files[x:x+100])

    dir_names.append(foldername)

    if not os.path.exists(foldername):
        os.mkdir(foldername)


for batch, folder in zip(batches, dir_names):
    for file in batch:
        filename = file.split("\\")[1]
        shutil.copy2(file, os.path.join(folder, filename))