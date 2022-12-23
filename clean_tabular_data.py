#%%
import pandas as pd
import numpy as np
import pathlib
import os
#%%
df_products = pd.read_csv('Products.csv', header=0, lineterminator='\n')
print(df_products.columns)
df_products.drop(columns=['Unnamed: 0'], inplace=True)

#%%

# pre-processing "price" column - removing £ sign and making numerical
df_products['price'] = df_products['price'].str.strip('£')
df_products['price'] = df_products['price'].apply(lambda x: x if ',' not in x else x.replace(',',''))
#%%
df_products['price'] = df_products['price'].astype('float64')
df_products.to_csv('Cleaned_products.csv')
#%%

# Exploring datasets
# %%
df_products.info()
# %%
# Exploring Images.csv and Products.csv datasets
df_images_ids = pd.read_csv('Images.csv', header=0, lineterminator='\n')
df_images_ids.drop(columns=['Unnamed: 0'], inplace=True)
df_images_ids.info()
print(f'There are {len(df_products)} entries in products dataset and {len(df_images_ids)} entries in images_to_ids dataset')
intersection = set(df_products['id']) & set(df_images_ids['product_id'])
print(f"There are {len(intersection)} images listed in both datasets")
# %%
df_images_ids.head()
# %%
#Loading names of images into a list 
imgdir_path = pathlib.Path('cleaned_images')
file_list = sorted([os.path.basename(str(path))[:-4] for path in imgdir_path.glob('*.jpg')])
print(file_list[0])

# %%
df_images_data = df_images_ids[df_images_ids['id'].isin(file_list)]

intersection = set(df_images_data['product_id']) & set(df_products['id'])
print(f"There are {len(file_list)} of images in total, \
        \nof which {len(df_images_data)} are listed in Images.csv file. \
        \n{len(intersection)} images have a corresponding entry in Image/product.csv")

# %%
file_list[:10]
# %%
combined_df = df_products.copy()
combined_df.rename(columns={'id': 'product_id'}, inplace=True)

combined_df.info()
# %%
combined_df = combined_df.merge(df_images_data, how='inner', on='product_id')
# %%
combined_df.info()
# %%
combined_df.rename(columns={'id': 'image_id'}, inplace=True)
combined_df.info()
# %%
# extracting main and subcategory
combined_df['main_category'] = combined_df['category'].apply(lambda x: x.split('/')[0].strip())
combined_df['sub_category'] = combined_df['category'].apply(lambda x: ('/').join(x.split('/')[1:]))
# %%
combined_df.head(5)
# %%
# checking labels are all unique
combined_df['main_category'].unique()
# %%
# changing 'main_category' type into a category
combined_df['main_category'] = combined_df['main_category'].astype('category')
# %%
combined_df.info()
# %%
print(combined_df['main_category'].unique().to_list())
combined_df.to_csv('combined_df.csv', index=False)
# %%
df = pd.read_csv('combined_df.csv',header=0, lineterminator='\n')


# %%
df.head(2)
# %%
