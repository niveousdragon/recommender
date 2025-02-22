import pyarrow.orc as orc
import pyarrow.parquet as pq
import os
import numpy as np

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

df0 = pd.read_csv("C:\\Projects\\temp\\full-scale v1\\file 2\\full-scale v1.0 p1 f2.csv")
df0['year'] = df0['date_actual'].apply(lambda date: int(date[:4]))
df0['month'] = df0['date_actual'].apply(lambda date: int(date[5:7]))
print(df0['month'].unique())