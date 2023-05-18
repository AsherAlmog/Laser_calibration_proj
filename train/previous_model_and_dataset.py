"""
# Load the saved numpy arrays
images = torch.load('images.pt')
labels = torch.load('labels.pt')
# images_tensor = torch.Tensor(images)
# labels_tensor = torch.Tensor(labels)


class SpeckleDataset(Dataset):
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
"""