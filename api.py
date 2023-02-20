import pickle
import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from pydantic import BaseModel
##############################################################
# TODO                                                       #
# Import your image processors here                 #
##############################################################
from image_processor import ImageProcessor


class ImageClassifier(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(ImageClassifier, self).__init__()

##############################################################
# TODO                                                       #
# Populate the __init__ method, so that it contains the same #
# structure as the model you used to train the image model   #
##############################################################
        
        self.decoder = decoder
        model_ft = models.resnet34(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, 1000)
        new_layer = torch.nn.Linear(1000, len(self.decoder))
        self.model = torch.nn.Sequential(model_ft, new_layer)

    def forward(self, image):
        x = self.model(image)
        return x

    def predict(self, image):
        with torch.no_grad():
            x = self.model(image)
            return x

    def predict_proba(self, image):
        x = self.predict(image)
        tensor_proba = torch.softmax(x, dim=1)
        max_proba, _ = torch.max(tensor_proba, dim=1)
        return round(max_proba.item(), 2)


    def predict_classes(self, image):
        with torch.no_grad():
            x = self.forward(image)
            predicted_class = torch.argmax(x, dim=1)
            label = self.decoder[predicted_class.item()]
            return label




# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str




try:
##############################################################
# TODO                                                       #
# Load the image model. Initialize a class that inherits from #
# nn.Module, and has the same structure as the image model   #
# you used for training it, and then load the weights in it. #
# Also, load the decoder dictionary that you saved as        #
# image_decoder.pkl                                          #
##############################################################
    with open("image_decoder.pkl", 'rb') as f:
        decoder = pickle.load(f)
    image_classifier = ImageClassifier(decoder)
    state_dict = torch.load('Resnet34_finetune_all_layers_lr_0.001_mom_0.8_acc__0.58.pt', map_location=torch.device('cpu'))
    image_classifier.model.load_state_dict(state_dict)

    # path = "/cleaned_images"
    # image = "/0a1baaa8-4556-4e07-a486-599c05cce76c.jpg"
    # image = Image.open(os.getcwd() + path + image)
    # image_processor = ImageProcessor(image_classifier.model)
    # model_input = image_processor.process(image)
    # image_classifier.predict_classes(model_input)
    # print(f" probability is {image_classifier.predict_proba(model_input)}")

except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")





try:
##############################################################
# TODO                                                       #
# Initialize the image processor that you will use to process#
# the text that you users will send to your API              #
##############################################################
    pass
except:
    raise OSError("No Image processor found. Check that you have the encoder and the model in the correct location")

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}


  
  
@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)

    ##############################################################
    # TODO                                                       #
    # Process the input and use it as input for the image model  #
    # image.file is the image that the user sent to your API     #
    # Apply the corresponding methods to compute the category    #
    # and the probabilities                                      #
    ##############################################################
    image_processor = ImageProcessor(image_classifier.model)
    model_input = image_processor.process(pil_image)
    image_classifier.predict_classes(model_input)
    return JSONResponse(content={
    "Category": image_classifier.predict_classes(model_input), # Return the category here
    "Probability": image_classifier.predict_proba(model_input) # Return a list or dict of probabilities here
        })
  

    
    
if __name__ == '__main__':
    uvicorn.run("api:app", host="0.0.0.0", port=8080)
