# Facebook Marketplace Recommendation Ranking System

Facebook Marketplace is a platform for buying and selling products on Facebook. This is an implementation of the system behind the marketplace, which uses AI to recommend the most relevant listings based on a personalised search query.

## Milestone 1

- Data was sourced from the EC2 instance and S3 bucket that contains: 
  -  two CSV files, and one folder. The two csv files are the dataset, and the folder contains the images corresponding to the products that you see in the dataset.
  -  The tabular dataset contains information about the listing, including its price, location, and description.
- Data cleaning of the tabular dataset included:
  -  all null values removed
  -  prices converted into a numerical format
  -  each image id in the ```Images.csv``` file is assigned a class ('main_category') extracted from the 'category' column of the ```products.csv``` file. 
- Created ```combined_df``` - an interscection between ```Images.csv``` and ```Products.csv```.
- Images dataset cleaning: each  image is downsized to 512x512
  
```python
def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size)) 
    new_im.paste(im, box=((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im
```

## Milestone 2 - Created a classification model by fine tuning a pre-trained Resnet model
- File: fb_marketplace.ipynb
- Created a train/valid/test PyTorch Datasets from cleaned image files saved locally by passing randomly generated indices for each data split.
  - As part of the class ```get_labels``` method saved an image decoder (a dictionary mapping label to class) to pkl.
  -   The ```__getitem__``` method in the ImageDataset loads the raw content:
    -  images from the cleaned files and labels obtained from the ```combined_df``` created in Milestone 1 and an image decoder
    -  transforms images by performing data augmentation, decodes it into tensor and normalises it:

```python
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize([256, 256]),
    transforms.GaussianBlur(kernel_size=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```
   - The exmaples and their classes:
![image](https://drive.google.com/uc?export=view&id=10nuG4gOf2w1JJVRHU2xexJGDCiNO9nr4)

- Used Resnet34 as a feature extractor and trained the last FC layer (1000,13) on the training dataset. Performed hyperparameter search across: learning rate, number of layers, momentum and number of epochs. Best results were achieved with lr = 0.001 / momentum = 0.8 / epochs = 3 / layers = 34


![image](https://drive.google.com/uc?export=view&id=10Y_GtOmpRFQM3qtMstvoRJOSXAn5ximD)

- Accuracy on the validation dataset increases to 58% when training a pre-trained Resnet34 model (same hyperparams as before) end-to-end. Accuarcy on the held-out test dataset is 58.3%.


![image](https://drive.google.com/uc?export=view&id=1fiEujhwxppJRl20kaCT2NTsEXcz3EBb3)

![image](https://drive.google.com/uc?export=view&id=1m3KCrKakY52JiuCuFQhjq26I8mHjzqYV)


- Obtained a feature extractor model by removing the last FC layer of the best perfoming pretrained classification model. Saved the final model weights: ```image_model.pt```
- Created an image processor script that takes an image/tensor and applies the transformation needed to be fed to the model:
```python
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
 ```

## Milestone 3 Created the search index using FAISS 
- Using the feature extraction model from the Milestone 2 - extracted feature embeddings for every image in the training dataset and saved to a JSON file where key is the image id and value is a 1000 dim tensor.
- Implemented FAISS model which takes a torch tensor as an input and searches for the k most similar indices amongst embedded labeled data.
```python
index = faiss.IndexFlatL2(1000)
import json
f = open(project_directory + '/image_embeddings.json')
data = json.load(f)
embeddings = [np.array(emb) for emb in data.values()]
embeddings = np.array(embeddings)
embeddings = np.float32(embeddings)

xq = np.ones((2,1000))
k = 4  # we want 4 similar vectors
D, I = index.search(np.float32(xq), k) # actual search
```

## Milestone 4 - Setup and deployed the API
- Load ```image_classifier``` class by loading the model weights into the model infrastructure.
- Build FastApi that will facilitate image classification 
```python 
@app.post('/predict/image')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)
    image_processor = ImageProcessor(image_classifier.model)
    model_input = image_processor.process(pil_image)
    image_classifier.predict_classes(model_input)
    return JSONResponse(content={
    "Category": image_classifier.predict_classes(model_input), 
    "Probability": image_classifier.predict_proba(model_input) 
        })
```
- Built a docker image and pushed it to Docker hub
```bash
docker build -t kosy9/fb_image_classification:latest .
docker run -p 8080:8080 -it kosy9/fb_image_classification:latest
```
- Created an EC2 instance and cloned git repository into it
```bash
ssh -i [xxxxxxxxxx].pem ec2-user@ec2-34-244-229-70.eu-west-1.compute.amazonaws.com
```
- scp ’ed the model into EC2 from my local machine
```bash
scp -i [pem file path] [file from local machine] ec2-user@instance(Punlic IPv4 address): [path within AWS]
```
- added ports to the security group of the EC@ instance:
  - Security group -> Edit  inbound rules -> Add rule:
      Type: Custom TCP
      Port range: 8080
      Source: Anywhere-IPv4 
- ran docker image inside the EC2 instance 
```bash 
sudo chmod 666 /var/run/docker.sock
docker login
sudo docker run -p 8080:8080 -it kosy9/fb_image_classification
```
- Test the API 
  - in the browser
```bash
http://<public IP>:8080/docs
http://ec2-3-250-17-102.eu-west-1.compute.amazonaws.com:8080/docs
```


![image](https://drive.google.com/uc?export=view&id=1rULhUdFuex6ZCh230ty5lx-dS6PIuof2)

  -  in Python:
```
import requests
files = {'image': open('/Users/mariakosyuchenko/AI_Core/facebook-marketplaces-recommendation-ranking-system/cleaned_images/0a1baaa8-4556-4e07-a486-599c05cce76c.jpg','rb')}
r = requests.post('http://3.250.17.102:8080/predict/image', files=files)
print(r.json())
```
