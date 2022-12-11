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
        imgdir_path = pathlib.Path('images')
        file_list = sorted([os.path.basename(str(path))[:-4] for path in imgdir_path.glob('*.jpg')])
        df = pd.read_csv("combined_df.csv", header=0, lineterminator='\n').sort_values(by='image_id')
        df = pd.DataFrame(file_list, columns=['image_id']).merge(df, how='inner', on='image_id')       
        return file_list, df

    def _get_labels(self):
        # print(sorted(list(self.df['main_category'].unique())))
        self.classes =sorted(list(self.df['main_category'].unique()))
        self.idx_to_class = {i: j for i, j in enumerate(self.classes)}
        self.class_to_idx = {value: key for key, value in self.idx_to_class.items()}

    def __getitem__(self, index):
        path = os.path.join('images/', str(self.file_list[index]) + ".jpg")
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

image_dataset[0][0].shape
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

