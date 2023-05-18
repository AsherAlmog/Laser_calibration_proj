# import cv2 as cv
import matplotlib.pyplot as plt
import torch
from torch import nn
# import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from torchvision import transforms
import torchvision.models as models
# from torchvision.io import read_image
from PIL import Image
import glob


class SpeckleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self._extract_label(image_path)
        return image, label

    def _get_image_paths(self):
        image_paths = []
        subfolders = glob.glob(os.path.join(self.data_dir, 'iter*'))

        for subfolder in subfolders:
            images = glob.glob(os.path.join(subfolder, '*.jpg'))
            image_paths.extend(images)

        return image_paths

    def _extract_label(self, image_path):
        # Extract label from the image file name or path
        # Modify this method based on the naming convention of your image files
        filename = os.path.basename(image_path)
        label = filename.split('.')[0].split(',')
        label = [int(l) for l in label]
        label = torch.tensor(label)
        label = label.float()
        return label


data_dir = 'E:/speckles_pic'

# Define the data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

debug_t = transforms.Compose([
    transforms.ToTensor(),
])
# Create an instance of the custom dataset
#laser_dataset = SpeckleDataset(images, labels, None)
laser_dataset = SpeckleDataset(data_dir, transform=debug_t)

# now we divide the dataset into test and training
training_percentage = 0.8
train_size = int(training_percentage * len(laser_dataset))
test_size = len(laser_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(laser_dataset, [train_size, test_size])
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# plotting an image for example:
first_image, first_label = next(iter(train_dataloader))

# Convert the image tensor to a NumPy array and transpose the dimensions
first_image = first_image[0].numpy().transpose(1, 2, 0)

# Display an image if wanted
plot_first_img_flag = 0
if plot_first_img_flag:
    plt.imshow(first_image)
    plt.title(f"Label: {first_label}")
    plt.axis('off')
    plt.show()


learning_rate = 0.001
# model = NeuralNetwork()

# Define the ResNet50 model with weights from ImageNet
output_size = 3
model = models.resnet50(weights='IMAGENET1K_V1')
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, output_size)  # Set the output layer to have 3 units

# Freeze the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Set the output layer to have requires_grad=True
for param in model.fc.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
model.to(device)  # Move the model to the device
print(f"working with device: {device}")
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
epochs = 30
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
plt.figure()
plt.plot(loss_lst)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("loss graph")
plt.savefig('loss_graph.png')
plt.show()
