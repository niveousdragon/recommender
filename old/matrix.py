import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import os
import tqdm
import json
import gc

def construct_sparse_user_item_matrix(transactions_df):
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

    # Map users and items to their respective indices
    transactions_df['human_idx'] = transactions_df['human'].map(user_mapping)
    transactions_df['cat_idx'] = transactions_df['top_cat_ind'].astype(str).map(item_mapping)

    # Create a sparse matrix using scipy's csr_matrix
    row_indices = transactions_df['human_idx']
    col_indices = transactions_df['cat_idx']
    data = [1] * len(transactions_df)  # Each transaction counts as 1 initially

    # Aggregate counts for duplicate (user, item) pairs
    sparse_matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(user_mapping), len(item_mapping))
    )

    # Sum duplicate entries to count transactions
    sparse_matrix.sum_duplicates()

    return sparse_matrix, user_mapping, item_mapping


ORC_INDS = [2]

all_dfs = []
for orc_ind in ORC_INDS:
    rpath = os.path.join(f'C:\\Projects\\temp\\full-scale v2 rubricator result\\file {orc_ind}')
    df_names = os.listdir(rpath)[:]

    for df_name in tqdm.tqdm(os.listdir(rpath)[:]):
        #df = pd.read_excel(os.path.join(rpath, df_name))
        df = pd.read_parquet(os.path.join(rpath, df_name), engine='pyarrow')
        #print(df.columns)
        all_dfs.append(df)

aggdf = pd.concat(all_dfs)

for df in all_dfs:
    del df
    gc.collect()

mat, um, im = construct_sparse_user_item_matrix(aggdf)

print(mat.shape)

# ==================================== SAVING =======================================
res_dir = '../orc2 results'
os.makedirs(res_dir, exist_ok=True)
postfix = 'orc2'


um_filename = os.path.join(res_dir, f'user mapping {postfix}.json')
# Save the dictionary to a JSON file
with open(um_filename, 'w', encoding='utf-8') as json_file:
    json.dump(um, json_file, ensure_ascii=False, indent=4)

im_filename = os.path.join(res_dir, f'cat mapping {postfix}.json')
# Save the dictionary to a JSON file
with open(im_filename, 'w', encoding='utf-8') as json_file:
    json.dump(im, json_file, ensure_ascii=False, indent=4)

mat_filename = os.path.join(res_dir, f'user cat matrix {postfix}.npz')
save_npz(mat_filename, mat)