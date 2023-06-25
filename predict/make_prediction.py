import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np
# Define these 3 constants based on your local directories:
images_dir = "/home/baralmog/specks/iter26/"  # don't forget the / at the end
img_name = "1,1,1.jpg"
model_path = "../train/model_params_0.pth"


# This function is reading an image and applies transforms to it
# In case of a problem, None is returned
def load_image(image_path, transform = None):
    try:
        image = Image.open(image_path)
        if transform:
            image = transform(image)
    except:
        image = None
    return image


image_path = images_dir+img_name
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Now when we have the image, we want to load the model:
# We use cpu for the predictions

# model = models.resnet18(pretrained=False)
# model.load_state_dict(torch.load(model_path)) #, map_location=torch.device('gpu')))


def make_prediction(model_path, image_path):
    model = torch.load(model_path, map_location='cpu')
    model.to('cpu')
    # If the model was originally trained using DataParallel or DistributedDataParallel
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module  # Get the underlying model

    # Move each parameter and buffer to CPU
    for parameter in model.parameters():
        parameter.data = parameter.data.to('cpu')
        if parameter._grad is not None:
            parameter._grad.data = parameter._grad.data.to('cpu')

    model.eval()
    image = load_image(image_path, transform)
    # Use the model to make predictions
    with torch.no_grad():
        input_image = torch.unsqueeze(image, 0)  # Add a batch dimension
        output = model(input_image)
    #output = np.copy(output.numpy())
    output = output.cpu().numpy()[0]
    return np.round(output)


print(make_prediction(model_path, image_path))