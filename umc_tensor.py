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


def preprocess_df(df):
    #print(df.head())
    df.dropna(inplace=True)
    df['date'] = df['date'].astype(str)
    df['year'] = df['date'].apply(lambda date: int(date[:4]))
    df['month'] = df['date'].apply(lambda date: int(date[5:7]))
    df.drop(['date'], axis=1, inplace=True)
    df.drop(['top_cat_sim'], axis=1, inplace=True)
    df['human'] = df['human'].apply(lambda v: v[:12])
    return df


res_dir = '10 orcs'
postfix = '10 orcs'
ORC_INDS = np.arange(10)

adf = None
for orc_ind in ORC_INDS:
    orc_parts = []
    rpath = os.path.join(f'C:\\Projects\\temp\\full-scale v2 rubricator result\\file {orc_ind}')
    df_names = os.listdir(rpath)[:]

    for df_name in tqdm.tqdm(os.listdir(rpath)[:]):
        df = pd.read_parquet(os.path.join(rpath, df_name), engine='pyarrow')
        pdf = preprocess_df(df)
        del df
        orc_parts.append(pdf)

    #create united orc df and add it to global one
    orcdf = pd.concat(orc_parts, ignore_index=True)
    if adf is None:
        adf = orcdf.copy()
    else:
        adf = pd.concat([adf, orcdf], ignore_index=True)

    # memory cleanup
    for pdf in orc_parts:
        del pdf
    del orcdf
    gc.collect() # force collecting garbage from a single .orc

print('calculating mapping for humans...')
#user_mapping, item_mapping = construct_user_item_mapping(adf)

adf['human_id'], _ = adf['human'].factorize()
adf.drop(['human'], axis=1, inplace=True)
#adf['human_id'] = adf['human'].astype('category').cat.codes
#----adf['human_id'] = adf['human'].map(user_mapping)
#print(adf['human_id'])

print('calculating mapping for cats...')
adf['cat_id'], _ = adf['top_cat_ind'].factorize()
adf.drop(['top_cat_ind'], axis=1, inplace=True)
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
'''
adf.dropna(inplace=True)
adf['date'] = adf['date'].astype(str)
adf['year'] = adf['date'].apply(lambda date: int(date[:4]))
adf['month'] = adf['date'].apply(lambda date: int(date[5:7]))
'''
adf['month_abs_number'] = (2024-adf['year'])*12 + adf['month']
adf['month_abs_number'] = adf['month_abs_number'] - min(adf['month_abs_number'])
adf.drop(['year', 'month'], axis=1, inplace=True)


print('constructing UMC tensor...')
# positive events
psubs1 = np.array(adf['human_id'].values, dtype=int)
adf.drop(['human_id'], axis=1, inplace=True)
psubs2 = np.array(adf['month_abs_number'].values, dtype=int)
adf.drop(['month_abs_number'], axis=1, inplace=True)
psubs3 = adf['cat_id'].values

#print(psubs1[:100], psubs2[:100], psubs3[:100])
print(min(psubs1), min(psubs2), min(psubs3))

nrows = len(adf)
del adf
gc.collect()

psubs = np.array([psubs1, psubs2, psubs3]).T  # Subscripts of +1.
print('indices shape:')
print(psubs.shape)

vals = np.ones(psubs.shape[0], dtype=int).reshape(-1,1)  # Vals is a column vector;
X = pyttb.sptensor.from_aggregator(psubs, vals)  # Sparse tensor

del psubs1, psubs2, psubs3, psubs, vals
gc.collect()

print(X.shape)

print('Saving UMC tensor...')
os.makedirs(res_dir, exist_ok=True)
np.savez(
    os.path.join(res_dir, f"sparse_user_month_cat_{postfix}.npz"),
    subs=X.subs,
    vals=X.vals,
    shape=X.shape
)


