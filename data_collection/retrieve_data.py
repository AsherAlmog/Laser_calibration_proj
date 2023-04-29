import os
import cv2
import numpy as np

root_dir = 'C:/Users/asher/PycharmProjects/Laser_calibration_proj/speckles_pic'  # directory where the images are saved
resolution = (50,50)
# Get a list of image filenames in the directory
img_filenames = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

# Load images and their labels
images = []
labels = []
for filename in img_filenames:
    # Load image
    img_path = os.path.join(root_dir, filename)
    img = cv2.imread(img_path)
    img = cv2.resize(img, resolution)
    images.append(img)

    # Extract label from filename
    label = filename.split('.')[0].split(',')
    label = [int(l) for l in label]
    labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Save the arrays
np.save('images.npy', images)
np.save('labels.npy', labels)

# Print the shapes of the arrays
print("Shape of images array:", images.shape)
print("Shape of labels array:", labels.shape)
