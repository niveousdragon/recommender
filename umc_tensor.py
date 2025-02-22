import gc
import os

import pandas as pd
import tqdm
import numpy as np
import pyttb
import json

def construct_user_item_mapping(transactions_df):
    """
    Constructs a sparse user-item matrix from a transactions DataFrame.

    Parameters:
        transactions_df (pd.DataFrame): A DataFrame containing at least two columns: 'user_id' and 'item_name'.

    Returns:
        csr_matrix: A sparse matrix where rows represent users, columns represent items,
                    and values represent the number of times a user bought an item.
        user_mapping (dict): A dictionary mapping user IDs to row indices in the matrix.
        item_mapping (dict): A dictionary mapping item names to column indices in the matrix.
    """
    # Ensure the input DataFrame has the required columns
    if 'human' not in transactions_df.columns or 'top_cat_ind' not in transactions_df.columns:
        raise ValueError("The input DataFrame must contain 'human' and 'top_cat_ind' columns.")

    # Create mappings for users and items to unique indices

    user_mapping = {user_id: idx for idx, user_id in enumerate(transactions_df['human'].unique())}
    item_mapping = {str(item_name): idx for idx, item_name in enumerate(transactions_df['top_cat_ind'].unique())}

    return user_mapping, item_mapping


res_dir = '3 orcs'
postfix = '3 orcs'
ORC_INDS = [2,3,4]

adf = None
for orc_ind in ORC_INDS:
    orc_parts = []
    rpath = os.path.join(f'C:\\Projects\\temp\\full-scale v2 rubricator result\\file {orc_ind}')
    df_names = os.listdir(rpath)[:]

    for df_name in tqdm.tqdm(os.listdir(rpath)[:]):
        df = pd.read_parquet(os.path.join(rpath, df_name), engine='pyarrow')
        orc_parts.append(df)

    #create united orc df and add it to global one
    orcdf = pd.concat(orc_parts, ignore_index=True)
    if adf is None:
        adf = orcdf.copy()
    else:
        adf = pd.concat([adf, orcdf], ignore_index=True)

    # memory cleanup
    for df in orc_parts:
        del df
    del orcdf
    gc.collect() # force collecting garbage from a single .orc

print('calculating mapping for humans...')
#user_mapping, item_mapping = construct_user_item_mapping(adf)

adf['human_id'], _ = adf['human'].factorize()
#adf['human_id'] = adf['human'].astype('category').cat.codes
#----adf['human_id'] = adf['human'].map(user_mapping)
#print(adf['human_id'])

print('calculating mapping for cats...')
adf['cat_id'], _ = adf['top_cat_ind'].factorize()
#adf['cat_id'] = adf['top_cat_ind'].astype('category').cat.codes
#-----adf['cat_id'] = adf['top_cat_ind'].map(item_mapping)
#print(adf['cat_id'])

print('processing month info...')
# Convert the 'date' column to datetime format
#adf['date'] = pd.to_datetime(adf['date'])

# Extract the month and add it as a new column
#adf['month'] = adf['date'].dt.month

# Extract the year and add it as a new column
#adf['year'] = adf['date'].dt.year

adf.dropna(inplace=True)
adf['date'] = adf['date'].astype(str)
adf['year'] = adf['date'].apply(lambda date: int(date[:4]))
adf['month'] = adf['date'].apply(lambda date: int(date[5:7]))

adf['month_abs_number'] = (2024-adf['year'])*12 + adf['month']
adf['month_abs_number'] = adf['month_abs_number'] - min(adf['month_abs_number'])

print('constructing UMC tensor...')
# positive events
psubs1 = np.array(adf['human_id'].values, dtype=int)
psubs2 = np.array(adf['month_abs_number'].values, dtype=int)
psubs3 = adf['cat_id'].values

#print(psubs1[:100], psubs2[:100], psubs3[:100])
print(min(psubs1), min(psubs2), min(psubs3))
psubs = np.array([psubs1, psubs2, psubs3]).T  # Subscripts of +1.
vals = np.ones(len(adf), dtype=int).reshape(-1,1)  # Vals is a column vector;
X = pyttb.sptensor.from_aggregator(psubs, vals)  # Sparse tensor

print(X.shape)
'''
S = X.collapse(dims=np.array([2])).double() # collapse over categories
h_nonempty_inds, m_nonempty_inds = np.where(S != 0) # nonempty user-month pairs
'''

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

os.makedirs(res_dir, exist_ok=True)
np.savez(
    os.path.join(res_dir, f"sparse_user_month_cat_{postfix}.npz"),
    subs=X.subs,
    vals=X.vals,
    shape=X.shape
)


