#%%
import pandas as pd
import numpy as np

#%%
df = pd.read_csv('Products.csv', header=0, lineterminator='\n')
print(df.columns)
df.drop(columns=['Unnamed: 0'], inplace=True)

#%%

# pre-processing "price" column - removing £ sign and making numerical
df['price'] = df['price'].str.strip('£')
df['price'] = df['price'].apply(lambda x: x if ',' not in x else x.replace(',',''))
#%%
df['price'] = df['price'].astype('float64')

#%%
df.dtypes
# %%
