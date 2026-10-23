import torch
from matplotlib import pyplot as plt
from torchvision import utils

from shared import organelle_detector

file = organelle_detector.clean_data(
    "../data/Dataset01/raw/Mouse192", "../data/Dataset01/clean"
)

# Try creating an Image Dataset
ds = organelle_detector.TEMImageDataset(file)

print(len(ds))

image, target = ds[1]

print(image.shape)
# print(target)
# print(image.dtype)

# Normalize the image
image_normalized = image / 65535.0

image_uint8 = (image_normalized * 255).to(torch.uint8)

img_with_boxes = utils.draw_bounding_boxes(
    image_uint8, target["boxes"], colors="yellow", width=3
)

print(target["boxes"])

plt_img = img_with_boxes.permute(1, 2, 0)

# 2. Plot the image
plt.imshow(plt_img)
plt.axis("off")  # Optional: Hides the pixel coordinate axes
plt.show()
