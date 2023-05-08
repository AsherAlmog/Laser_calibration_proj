import zipfile
# import csv
import torch.nn.functional as F
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import os
import torchvision.transforms as T
from torchvision import datasets, transforms
import torchvision.models as models
from torchvision.transforms import ToTensor, Lambda
import math
# from torchvision.io import read_image
from PIL import Image

# Load the saved numpy arrays
images = torch.load('images.pt')
labels = torch.load('labels.pt')
# images_tensor = torch.Tensor(images)
# labels_tensor = torch.Tensor(labels)


# Define a custom dataset class
class SpecklesDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.Tensor(label)
        image = image.to(torch.float)
        return image, label


# Define the data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

debug_t = transforms.Compose([transforms.Resize((20, 20))])
# Create an instance of the custom dataset
laser_dataset = SpecklesDataset(images, labels, None)

# Define the data loader
# batch_size = 32
# laser_dataloader = DataLoader(laser_dataset, batch_size=batch_size, shuffle=True)


# change first size to (1280,960)

# define necessary sizes for the hidden layers
depths_list = [3, 6, 9]  # initial depth and the depths after each conolution
kernel_sizes = [5,3]  # kernel sizes for the convolution layers
pooling_sizes = [2,2]  # max pooling size after each layer
convolutions_sizes = [(20,20)]  # the size of the image before each convolution layer


def calc_size_next_layer(prev_size, kernel_size, pooling_size):
    size = int((prev_size-kernel_size+1)/pooling_size)
    return size


for i in range(len(depths_list)-1):
    size_0 = calc_size_next_layer(convolutions_sizes[i][0],kernel_sizes[i], pooling_sizes[i])
    size_1 = calc_size_next_layer(convolutions_sizes[i][1], kernel_sizes[i], pooling_sizes[i])
    size = (size_0, size_1)
    convolutions_sizes.append(size)

l = len(convolutions_sizes)
first_fc_input_size = int((convolutions_sizes[l-1][0]*convolutions_sizes[l-1][1])*depths_list[-1])
output_size = 3

print(f" expected size is {first_fc_input_size}")
fc_layers_sizes = [first_fc_input_size, 15, 5, output_size]


# now we divide the dataset into test and training
training_percentage = 0.8
train_size = int(training_percentage * len(laser_dataset))
test_size = len(laser_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(laser_dataset, [train_size, test_size])

# plotting an image for example:
# X1, y1 = train_dataset[0]
# plt.imshow(X1, cmap='gray')
# plt.show()
# print(f"training samples amount is {len(train_dataset)}")


batch_size = 2

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(depths_list[0], depths_list[1], kernel_sizes[0])
        self.pool1 = nn.MaxPool2d(pooling_sizes[0])
        self.conv2 = nn.Conv2d(depths_list[1], depths_list[2], kernel_sizes[1])
        self.pool2 = nn.MaxPool2d(pooling_sizes[1])
        self.fc_layer = nn.Sequential(nn.Linear(fc_layers_sizes[0], fc_layers_sizes[1]),
                                                    nn.LeakyReLU(), nn.Dropout(0.2), nn.Linear(fc_layers_sizes[1], fc_layers_sizes[2]),
                                                    nn.LeakyReLU(), nn.Dropout(0.2), nn.Linear(fc_layers_sizes[2], output_size))

    def forward(self, x):
        batch_size = x.size(0)
        #print(f"x is with size {x.shape}")
        x = self.conv1(x)
        #print(f"x is with size {x.shape}")
        x = self.pool1(x)
        #print(f"x is with size {x.shape}")
        x = self.conv2(x)
        #print(f"x is with size {x.shape}")
        x = self.pool2(x)
        #print(f"x is with size {x.shape}")
        x = x.view(batch_size, -1)  # flatten the vector but keep the batch size
        # print(f" actual size is {x.shape}")

        #print(f"x is with size {x.shape}")
        y = self.fc_layer(x)  # activate the Neural Network
        return y


learning_rate = 0.001
# model = NeuralNetwork()

# Define the ResNet50 model with weights from ImageNet
model = models.resnet50(weights='IMAGENET1K_V1')
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, output_size)  # Set the output layer to have 3 units

# Freeze the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Set the output layer to have requires_grad=True
for param in model.fc.parameters():
    param.requires_grad = True


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
epochs = 50
loss_lst = []

def train(train_dataloader, loss_fn, optimizer):
    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        # Compute prediction and loss
        # X = torch.Tensor(X)
        # y = torch.Tensor(y)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Log training loss
    train_loss /= num_batches
    print(f"Training Error: \n Avg loss: {train_loss:>8f} \n")


def test(test_dataloader, loss_fn, loss_lst):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            test_loss += loss.item()

        test_loss /= num_batches
        loss_lst.append(test_loss)
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


for epoch in range(epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train(train_dataloader, loss_fn, optimizer)
    test(test_dataloader, loss_fn, loss_lst)

torch.save(model.state_dict(), 'model_params.pth')
# plt.subplot(2, 1, 1)
plt.figure()
plt.plot(loss_lst)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("loss graph")
plt.show()
