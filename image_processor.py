import torch
import os
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from PIL import Image

class ImageProcessor:
    def __init__(self, model=None):
        self.model = model
        self.transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def process(self, input):
        if torch.is_tensor(input):
            img_tensor = input.unsqueeze(0)
        else:
            img_tensor = self.transform(input).unsqueeze(0)
        # return self.model(img_tensor).detach().squeeze(0).numpy()
        return img_tensor

# if __name__ == "__main__":
#     pass
#     # path = "/cleaned_images"
#     # # image = "/0a1baaa8-4556-4e07-a486-599c05cce76c.jpg"
#     # # image = Image.open(os.getcwd() + path + image)
#     # processor = ImageProcessor()
#     # print(processor.process(image).shape)