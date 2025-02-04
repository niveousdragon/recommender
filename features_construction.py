import os
import pandas as pd
import tqdm
import numpy as np
import pyttb
import json

rpath = os.path.join('C:\\Projects\\temp\\mass test result 2')
df_names = os.listdir(rpath)[:]

res_dir = '75M results'
postfix = '75M'
im_filename = os.path.join(res_dir, f'cat mapping {postfix}.json')

# load category mapping
if os.path.exists(im_filename):
    with open(im_filename, 'r', encoding='utf-8') as json_file:
        item_mapping = json.load(json_file)
else:
    raise Exception('Item mapping not found')

all_dfs = []
for df_name in tqdm.tqdm(os.listdir(rpath)[:75]):
    df = pd.read_excel(os.path.join(rpath, df_name))
    all_dfs.append(df)
adf = pd.concat(all_dfs)

# Convert the 'date' column to datetime format
adf['date'] = pd.to_datetime(adf['date'])

# Extract the month and add it as a new column
adf['month'] = adf['date'].dt.month

# Extract the year and add it as a new column
adf['year'] = adf['date'].dt.year

adf['month_abs_number'] = (2024-adf['year'])*12 + adf['month']
adf['month_abs_number'] = adf['month_abs_number'] - min(adf['month_abs_number'])
adf['cat_id'] = adf['top_cat'].map(item_mapping)
#print(item_mapping)

subs1 = np.arange(len(adf), dtype=int)
subs2 = adf['month_abs_number'].values
subs3 = adf['cat_id'].values

#print(subs1[:100], subs2[:100], subs3[:100])
subs = np.array([subs1, subs2, subs3]).T  # Subscripts of the nonzeros.
vals = np.ones(len(adf), dtype=int).reshape(-1,1)  # Vals is a column vector; values of the nonzeros.
X = pyttb.sptensor.from_aggregator(subs, vals)  # Sparse tensor

print(X.shape)
np.savez(
    os.path.join(res_dir, f"sparse_user_month_cat_{postfix}.npz"),
    subs=X.subs,
    vals=X.vals,
    shape=X.shape
)


