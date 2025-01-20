import pyorc
import os

from pandas.core.dtypes.common import ensure_int16
from pyorc.enums import StructRepr
import pandas as pd
import tqdm
import numpy as np

PATH = "C:\\Projects\\temp\\data\\"
SAVEPATH = "C:\\Projects\\temp\\mass test"
os.makedirs(SAVEPATH, exist_ok=True)

nchunks = 20
total_size = 20*1e6
chunk_size = total_size//nchunks

if __name__ == '__main__':

    all_files = os.listdir(PATH)
    print(all_files)

    save = True
    strs = []
    with open(os.path.join(PATH, all_files[0]), "rb") as f:
        reader = pyorc.Reader(f, struct_repr=StructRepr.DICT)
        #print(str(reader.schema))

        counter = 0
        i=0
        while i < total_size:
            data = next(reader)
            product = []
            if data['NER'] is not None:
                if len(data['NER']) != 0:
                    for part in data['NER']:
                        if part['ner'] == 'PRODUCT':
                            product.append(part['text'])

            data['product'] = product
            strs.append(data)

            if i % chunk_size == 0 and i!=0:
                part = int(i//chunk_size)
                print(f'part {part} completed')
                df = pd.DataFrame.from_records(strs)
                print(df)
                strs = []

                if save:
                    df.to_csv(os.path.join(SAVEPATH, f'fd mass test 1.0 p{part}.csv'))#, encoding='cp1251')

            i+=1
