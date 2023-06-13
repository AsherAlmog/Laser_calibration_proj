import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image

# Define these 3 constants based on your local directories:
images_dir = "E:/speckles_pic/"  # don't forget the / at the end
img_name = "test_img.png"
model_path = "model_params.pth"

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
image = load_image(image_path, transform)
# Now when we have the image, we want to load the model:
# We use cpu for the predictions

model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Use the model to make predictions
with torch.no_grad():
    output = model(image)
    x = 1