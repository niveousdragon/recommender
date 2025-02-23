import pyarrow.orc as orc
import pyarrow.parquet as pq
import os
import numpy as np
import tqdm

import pandas as pd

# Example DataFrame
data = {
    "value": ["A", "B", "A", "C", "B"]
}
df = pd.DataFrame(data)

# Convert 'value' column to categorical and get codes (IDs)
df['unique_id'] = df['value'].astype('category').cat.codes
df['unique_ids'], _ = df['value'].factorize()
print(df)

dfs = []
for _ in tqdm.tqdm(np.arange(70)):
    df0 = pd.read_csv(f"C:\\Projects\\temp\\full-scale v1\\file 2\\full-scale v1.0 p{_+1} f2.csv")
    df0['year'] = df0['date_actual'].apply(lambda date: int(date[:4]))
    df0['month'] = df0['date_actual'].apply(lambda date: int(date[5:7]))
    dfs.append(df0)
    df0['tr_human_id'] = df0['human_id'].apply(lambda v: v[:12])#v.split('-')[0])

adf = pd.concat(dfs)
print(len(adf['human_id'].unique()))
print(len(adf['tr_human_id'].unique()))