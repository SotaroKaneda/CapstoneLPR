import cv2
import os
import matplotlib.pyplot as plt


image_path = os.path.join(r"D:\v2x-11-30-data\crops\Train", "001315_1701100170740R03_372.png")
image = cv2.imread(image_path)
cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(cvt_image, cv2.COLOR_RGB2HSV)
# Histogram equalisation on the V-channel
img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])

# convert image back from HSV to RGB
eq_hist_image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

# remove noise
no_noise = cv2.fastNlMeansDenoisingColored(cvt_image,None,10,10,7,21)

plt.figure(1)
plt.imshow(cvt_image)

plt.figure(2)
plt.imshow(~cvt_image)

plt.figure(3)
plt.imshow(eq_hist_image)

# plt.figure(4)
# plt.imshow(no_noise)

plt.show()
