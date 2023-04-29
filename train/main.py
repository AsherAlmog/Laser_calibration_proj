import zipfile
import csv
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
from torch.utils.data import DataLoader
import os
import torchvision.transforms as T
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import math
from torchvision.io import read_image

def calc_size_next_layer(prev_size, kernel_size, pooling_size):
    size = int((prev_size-kernel_size+1)/pooling_size)
    return size




data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])  # extract the image name from the csv file
        image = read_image(img_path)
        #image = image.numpy()
        #print(f" type {image.dtype}")
        label = self.img_labels.iloc[idx, 1]   # extract the image label from the csv file
        # now we turn the label into a 4-dim vector
        lst = label.split(',')
        for k in range(len(lst)):
            lst[k] = int(lst[k])
            lst[k] = float(lst[k])
        tensor_label = torch.tensor(lst)
        if self.transform:
            image = self.transform(image/255)
        if self.target_transform:
            label = self.target_transform(label)
        return image, tensor_label

laser_dataset = CustomImageDataset(annotations_file="laser_csv.csv", img_dir="Laser_Pics", transform=data_transforms['train'])



# this function gets a 3-D image as tensor from the dataset and plot it with plt.imshow
def show_image(x):
    x_new = x.numpy().transpose((1,2,0))
    print(f"X shape is {x_new.shape}")
    plt.imshow(x_new)
    plt.show()

#change first size to (1280,960)

# define necessary sizes for the hidden layers
depths_list = [3, 6, 9]  #initial depth and the depths after each conolution
kernel_sizes = [5,3]  #kernel sizes for the convolution layers
pooling_sizes = [2,2]  #max pooling size after each layer
convolutions_sizes = [(224,224)] #the size of the image before each convolution layer

for i in range(len(depths_list)-1):
    size_0 = calc_size_next_layer(convolutions_sizes[i][0],kernel_sizes[i], pooling_sizes[i])
    size_1 = calc_size_next_layer(convolutions_sizes[i][1], kernel_sizes[i], pooling_sizes[i])
    size = (size_0, size_1)
    convolutions_sizes.append(size)

l = len(convolutions_sizes)
first_fc_input_size = int((convolutions_sizes[l-1][0]*convolutions_sizes[l-1][1])*depths_list[-1])
output_size = 4

print(f" expected size is {first_fc_input_size}")
fc_layers_sizes = [first_fc_input_size, 15, 5, output_size]




# now we divide the dataset into test and training
training_percentage = 0.8
train_size = int(training_percentage * len(laser_dataset))
test_size = len(laser_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(laser_dataset, [train_size, test_size])
X1, y1 = train_dataset[0]
print(f"training samples amount is {len(train_dataset)}")


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
model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()  # already applies softmax at the end
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
epochs = 50
loss_lst = []
acc_lst = []


def train(train_dataloader, loss_fn, optimizer):
    size = len(train_dataloader.dataset)
    for batch, (X, y) in enumerate(train_dataloader):
        y_hat = model(X)
        # print(f"y_hat's shape:{y_hat.shape} and y's shape: {y.shape}")
        # print(f"y_hat is:{y_hat} and y is: {y}")
        # calculate Cross-Entropy loss
        loss = loss_fn(y_hat, y)

        # backpropagation
        loss.backward()

        # update model weights
        optimizer.step()

        # clean gradients
        optimizer.zero_grad()

        if batch % 2 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    scheduler.step()

def test(test_dataloader, loss_fn, optimizer, loss_lst, acc_lst):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_dataloader:
            y_hat = model(X)
            loss = loss_fn(y_hat, y).item()
            test_loss += loss
            pred = y_hat.argmax(1)
            actual = y.argmax(1)
            correct += (pred == actual).type(torch.float).sum().item()  # sum the number of correct predictions

    test_loss /= num_batches
    correct /= size
    acc = 100 * correct
    loss_lst.append(test_loss)
    acc_lst.append(acc)
    print(f"Test Error: \n Accuracy: {(acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for epoch in range(epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train(train_dataloader, loss_fn, optimizer)
    test(test_dataloader, loss_fn, optimizer, loss_lst, acc_lst)

plt.subplot(2, 1, 1)
plt.plot(loss_lst)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("loss graph")

plt.subplot(2, 1, 2)
plt.plot(acc_lst)
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title("acc graph")
plt.show()






""""




X1, y1 = face_dataset[0]
# print(f"y1 is {y1}")
X2, y2 = face_dataset[150]

show_image(X2)

# now we divide the dataset into test and training
training_percentage = 0.8
train_size = int(training_percentage * len(face_dataset))
test_size = len(face_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(face_dataset, [train_size, test_size])
X1, y1 = train_dataset[0]
print(f"y1 is {y1} and y1 type is {type(y1)}")


batch_size = 2
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(train_dataloader)







initial_depth = 3
secondary_depth = 16
kernel_size_1 = 5
kernel_size_2 = 5
pooling_size = 2
size_after_1 = int((img_reshape_size-kernel_size_1+1)/pooling_size)
size_after_2 = int((size_after_1-kernel_size_2+1)/pooling_size)
first_fc_input_size = int((size_after_2**2)*secondary_depth)
"""""


