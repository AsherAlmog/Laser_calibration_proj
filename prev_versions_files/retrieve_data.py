import os
import cv2
import numpy as np
import torch
from torchvision.io import read_image
from torchvision import transforms
root_dir = 'C:/Users/asher/PycharmProjects/Laser_calibration_proj/speckles_pic'  # directory where the images are saved
resolution = (1280, 960)  # change the resolution to (1280,960)
# Get a list of image filenames in the directory
img_filenames = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

# Load images and their labels
images = []
labels = []
for filename in img_filenames:
    # Load image
    img_path = os.path.join(root_dir, filename)
    img = read_image(img_path)
    T = transforms.Resize(resolution)
    img = T(img)
    images.append(img)

    # Extract label from filename
    label = filename.split('.')[0].split(',')
    label = [int(l) for l in label]
    labels.append(label)


# Save the arrays
torch.save(images, '../train/images.pt')
torch.save(labels, '../train/labels.pt')
