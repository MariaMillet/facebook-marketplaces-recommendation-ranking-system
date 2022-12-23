#%%
import torch
import torchvision.transforms as transforms
import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
from torch.utils.data import DataLoader, Dataset
# %%
transform = transforms.Compose([
    transforms.ToTensor()
])
class ImageDataset(Dataset):
    def __init__(self, transform=None):
        self.file_list, self.df = self._get_file_list()
        self.transform = transform
        self._get_labels()
    

    def _get_file_list(self):
        imgdir_path = pathlib.Path('cleaned_images')
        file_list = sorted([os.path.basename(str(path))[:-4] for path in imgdir_path.glob('*.jpg')])
        df = pd.read_csv("combined_df.csv", header=0, lineterminator='\n').sort_values(by='image_id')
        print(df.head(2))
        # print(pd.DataFrame(file_list, columns=['image_id']).head(1))
        df = pd.DataFrame(file_list, columns=['image_id']).merge(df, how='inner', on='image_id')       
        return file_list, df

    def _get_labels(self):
        # print(sorted(list(self.df['main_category'].unique())))
        self.classes =sorted(list(self.df['main_category'].unique()))
        self.idx_to_class = {i: j for i, j in enumerate(self.classes)}
        self.class_to_idx = {value: key for key, value in self.idx_to_class.items()}

    def __getitem__(self, index):
        path = os.path.join('cleaned_images/', str(self.file_list[index]) + ".jpg")
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        class_label = self.df[self.df['image_id'] == self.file_list[index]]['main_category']
        label = self.class_to_idx[class_label.item()]
        return img, label
    
    def __len__(self):
        return len(self.file_list)


image_dataset = ImageDataset(transform)
# %%

print(image_dataset[0][0].shape)
image_dataset.df.head()
# %%
fig = plt.figure(figsize=(10,6))

for i, example in enumerate(image_dataset):
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(example[0].numpy().transpose((1, 2, 0)))
    # print(image_dataset.idx_to_class[example[1]])
    ax.set_title(f"{image_dataset.idx_to_class[example[1]]}", size=15)
    if i == 5:
        break

plt.tight_layout()
plt.show()


# %%
class CNNModel(torch.nn.Module()):
    def __init__(self):
        super().__init__()
        #initial image = (3, 512, 512)
        self.conv1 = torch.nn.Sequential(
                     torch.nn.Conv2d(3, 32, kernel_size=3, padding=2),
                     torch.nn.ReLU(),
                     torch.nn.MaxPool2D(kernel_size=2),
                     torch.nn.Dropout(p=0.5)
        )
        #image = (32, 256, 256)
        self.conv2 = torch.nn.Sequential(
                     torch.nn.Conv2d(32, 64, kernel_size=3, padding=2),
                     torch.nn.ReLU(),
                     torch.nn.MaxPool2D(kernel_size=2),
                     torch.nn.Dropout(p=0.5)
        )
        #image = (64, 128, 128)
        self.conv3 = torch.nn.Sequential(
                     torch.nn.Conv2d(64, 128, kernel_size=3, padding=2),
                     torch.nn.ReLU(),
                     torch.nn.MaxPool2D(kernel_size=2),
                     torch.nn.Dropout(p=0.5)
        )
        #image = (128, 64, 64)
        self.conv4 = torch.nn.Sequential(
                     torch.nn.Conv2d(128, 256, kernel_size=3, padding=2),
                     torch.nn.ReLU(),
                     torch.nn.MaxPool2D(kernel_size=2),
                     torch.nn.Dropout(p=0.5)
        )
        #image = (256, 32, 32)
        self.conv5 = torch.nn.Sequential(
                     torch.nn.Conv2d(256, 512, kernel_size=3, padding=2),
                     torch.nn.ReLU(),
                     torch.nn.MaxPool2D(kernel_size=2),
                     torch.nn.Dropout(p=0.5)
        )
        #image = (512, 16, 16)
        self.av_pool = torch.nn.AvgPool2d(kernel_size=16)
        # image = (512, 1, 1)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(512, 13)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.av_pool(x)
        x = self.flatten(x)
        x = self.linear(x)