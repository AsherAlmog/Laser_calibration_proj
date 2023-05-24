import cv2 as cv
import matplotlib.pyplot as plt
import torch
from torch import nn
# import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from torchvision import transforms
import torchvision.models as models
import torchvision.io as io
from PIL import Image
import glob
from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    batch = [data for data in batch if data[0] is not None]  # Remove None samples
    if len(batch) == 0:
        return None, None
    return default_collate(batch)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
class SpeckleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
        # image = image[0, :, :]
        except:
            return None, None
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


data_dir = '/home/baralmog/specks'

# Define the data transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)


learning_rate = 0.01
# model = NeuralNetwork()

# Define the ResNet50 model with weights from ImageNet
output_size = 3
# model = models.resnet50(weights='IMAGENET1K_V1')
pretrained_flag = 0
if pretrained_flag:
    pretrained = True
    requires_grad = False
else:
    pretrained = False
    requires_grad = True

model = models.resnet50(pretrained=pretrained)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, output_size)  # Set the output layer to have 3 units

# Freeze the pre-trained layers
for param in model.parameters():
    param.requires_grad = requires_grad

# Set the output layer to have requires_grad=True
if pretrained_flag:
    for param in model.fc.parameters():
        param.requires_grad = True




#model.to(device)  # Move the model to the device
device_ids = [0,1,2,3]
model = model.cuda()
model = nn.DataParallel(model, device_ids=device_ids)
print(f"working with device: {device}")
loss_fn = nn.MSELoss()# .to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)
epochs = 60
loss_lst = []


def train(train_dataloader, loss_fn, optimizer, device):
    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        if(X==None):
            continue
        # Compute prediction and loss
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
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


def test(test_dataloader, loss_fn, loss_lst, device):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            if (X==None):
                continue
            X = X.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            test_loss += loss.item()

        test_loss /= num_batches
        loss_lst.append(test_loss)
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


for epoch in range(epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    train(train_dataloader, loss_fn, optimizer, device)
    test(test_dataloader, loss_fn, loss_lst, device)

torch.save(model.state_dict(), 'model_params.pth')
plt.figure()
plt.plot(loss_lst)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("loss graph")
plt.savefig('loss_graph.png')
plt.show()
