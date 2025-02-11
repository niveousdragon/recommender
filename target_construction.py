import os
import pandas as pd
import tqdm
import numpy as np
import pyttb
import json

rpath = os.path.join('C:\\Projects\\temp\\mass test result 2')
#rpath = os.path.join('C:\\Projects\\temp\\full-scale v1 rubricator result\\file 2')
df_names = os.listdir(rpath)[:]

res_dir = '75M results'
postfix = '75M'
im_filename = os.path.join(res_dir, f'cat mapping {postfix}.json')
um_filename = os.path.join(res_dir, f'user mapping {postfix}.json')

# load category mapping
if os.path.exists(im_filename):
    with open(im_filename, 'r', encoding='utf-8') as json_file:
        item_mapping = json.load(json_file)
else:
    raise Exception('Item mapping not found')

ncat = len(item_mapping)

# load user mapping
if os.path.exists(um_filename):
    with open(um_filename, 'r', encoding='utf-8') as json_file:
        user_mapping = json.load(json_file)
else:
    raise Exception('Item mapping not found')

# load category mapping# load category mapping
# if os.path.exists(im_filename):
#     with open(im_filename, 'r', encoding='utf-8') as json_file:
#         item_mapping = json.load(json_file)
# else:
#     raise Exception('Item mapping not found')
if os.path.exists(im_filename):
    with open(im_filename, 'r', encoding='utf-8') as json_file:
        item_mapping = json.load(json_file)
else:
    raise Exception('Item mapping not found')

all_dfs = []
for df_name in tqdm.tqdm(os.listdir(rpath)[:]):
    df = pd.read_excel(os.path.join(rpath, df_name))
    all_dfs.append(df)
adf = pd.concat(all_dfs)
adf['human_id'] = adf['human'].map(user_mapping)

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

# positive events
psubs1 = np.array(adf['human_id'].values, dtype=int)
psubs2 = adf['month_abs_number'].values
psubs3 = adf['cat_id'].values

#print(psubs1[:100], psubs2[:100], psubs3[:100])
psubs = np.array([psubs1, psubs2, psubs3]).T  # Subscripts of +1.
vals = np.ones(len(adf), dtype=int).reshape(-1,1)  # Vals is a column vector;
X = pyttb.sptensor.from_aggregator(psubs, vals)  # Sparse tensor

print(X.shape)
S = X.collapse(dims=np.array([2])).double() # collapse over categories
h_nonempty_inds, m_nonempty_inds = np.where(S != 0) # nonempty user-month pairs

'''
# all events (mostly negative)
asubs1 = np.repeat(h_nonempty_inds, ncat)
asubs2 = np.repeat(m_nonempty_inds, ncat)
asubs3 = np.repeat(np.arange(ncat), len(h_nonempty_inds))

#print(nsubs1[:100], nsubs2[:100], nsubs3[:100])
asubs = np.array([asubs1, asubs2, asubs3]).T  # Subscripts of the +1 and -1.
all_vals = -np.ones((len(h_nonempty_inds), ncat)) # all values
all_vals[psubs1, psubs3] = 1 # return +1 to data

fvals = all_vals.ravel().reshape(-1,1)  # ravel returns row1-row2-row3...
Xf = pyttb.sptensor.from_aggregator(asubs, fvals)  # Sparse tensor

print(Xf.shape)
'''
np.savez(
    os.path.join(res_dir, f"sparse_user_month_cat_{postfix}.npz"),
    subs=X.subs,
    vals=X.vals,
    shape=X.shape
)


