import pyorc
import os
from pyorc.enums import StructRepr
import pandas as pd
import tqdm
import numpy as np

PATH = "C:\\Projects\\temp\\data"

if __name__ == '__main__':

    all_files = os.listdir(PATH)
    #print(all_files)

    strs = []
    with open(os.path.join(PATH, all_files[0]), "rb") as f:
        reader = pyorc.Reader(f, struct_repr=StructRepr.DICT)
        #print(str(reader.schema))

        for _ in tqdm.tqdm(np.arange(10000)):
            data = next(reader)
            #print(data)
            product = []
            if data['NER'] is not None:
                if len(data['NER']) != 0:
                    for part in data['NER']:
                        if part['ner'] == 'PRODUCT':
                            product.append(part['text'])

            data['product'] = product
            strs.append(data)

    df = pd.DataFrame.from_records(strs)
    print(df)

    save = 1
    if save:
        df.to_csv(os.path.join(PATH, 'test', 'test fd 3.csv'))#, encoding='cp1251')

