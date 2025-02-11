import pyarrow.orc as orc
import pyarrow.parquet as pq
import os
import numpy as np

'''
DATAPATH = "C:\\Projects\\temp\\data\\"
SAVEPATH = "C:\\Projects\\temp\\full-scale v2"
os.makedirs(SAVEPATH, exist_ok=True)

orc_indices = np.arange(3)
all_files = sorted(os.listdir(DATAPATH))

def process_ner(ner):
    product = []
    if ner is not None:
        if len(ner) != 0:
            for part in ner:
                if part['ner'] == 'PRODUCT':
                    product.append(part['text'])
    return product


if __name__ == '__main__':
    for orc_ind in orc_indices:

        # Read specific columns for better performance
        table = orc.read_table(os.path.join(DATAPATH, all_files[orc_ind]),
                               columns=['human_id', 'date_actual', 'NER'])

        products = table['NER'].to_pandas().apply(process_ner)
        table = table.append_column('product', products)
        table.drop(['NER'])

        os.makedirs(os.path.join(SAVEPATH, f'file {orc_ind}'), exist_ok=True)
        pq.write_table(table, os.path.join(SAVEPATH, f'file {orc_ind}', f'full-scale v2.0 f{orc_ind}.parquet'))
'''
arr = np.array([
    [1, 10, 100],
    [1, 10, 150],
    [2, 20, 200],
    [1, 30, 300],
    [2, 20, 400],
    [2, 20, 500]
])

# Values to search for in the first and second columns
values1 = np.array([1, 2])  # Length k
values2 = np.array([10, 20])  # Length k

# Vectorized search: Create a boolean mask for each condition
mask1 = arr[:, 0][:, None] == values1[None, :]  # Match values in the first column
mask2 = arr[:, 1][:, None] == values2[None, :]  # Match values in the second column

print(mask1.astype(int))
print(mask2.astype(int))
# Combine masks using logical AND to find rows matching both conditions
combined_mask = mask1 & mask2
print(combined_mask.astype(int))

# Extract indices where combined_mask is True
row_indices = np.where(combined_mask)[0]
# Extract corresponding values from the third column
result = arr[row_indices, 2]

print("Result:", result)
counts = np.sum(combined_mask.astype(int), axis=0)
borders = np.zeros(3)
borders[1:] = np.cumsum(counts)
borders = borders.astype(int)

print([result[borders[i]: borders[i+1]] for i in range(2)])

