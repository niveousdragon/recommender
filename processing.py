import pyorc
import os

from pyorc.enums import StructRepr
import pandas as pd
import tqdm
import numpy as np
from multiprocessing import Pool

def do_ind(orc_ind):

    save = True
    strs = []
    with open(os.path.join(DATAPATH, all_files[orc_ind]), "rb") as f:
        reader = pyorc.Reader(f, struct_repr=StructRepr.DICT)
        #print(str(reader.schema))

        counter = 0
        i=0
        while i < total_size_cap:
            try:
                data = next(reader)
                product = []
                if data['NER'] is not None:
                    if len(data['NER']) != 0:
                        for part in data['NER']:
                            if part['ner'] == 'PRODUCT':
                                product.append(part['text'])

                light_data = {key: data[key] for key in ['human_id', 'date_actual', 'sum']}
                light_data['product'] = product
                strs.append(light_data)

                if i % chunk_size == 0 and i!=0:
                    part = int(i//chunk_size)
                    print(f'part {part} of file {orc_ind} completed')
                    df = pd.DataFrame.from_records(strs)
                    #print(df)
                    strs = []

                    if save:
                        os.makedirs(os.path.join(SAVEPATH, f'file {orc_ind}'), exist_ok=True)
                        df.to_csv(os.path.join(SAVEPATH, f'file {orc_ind}', f'full-scale v1.0 p{part} f{orc_ind}.csv'))#, encoding='cp1251')

                i+=1

            except StopIteration:
                print(f'file {orc_ind} finished')
                break


DATAPATH = "C:\\Projects\\temp\\data\\"
#DATAPATH = "/usrshome/kpolovnikov/data/"
SAVEPATH = "C:\\Projects\\temp\\full-scale v1"
#SAVEPATH = "/usrshome/kpolovnikov/full-scale v1"

os.makedirs(SAVEPATH, exist_ok=True)

all_files = sorted(os.listdir(DATAPATH))
#print(all_files)

total_size_cap = 100*1e6 # max number of lines from 1 file, has no effect if large enough, useful for experiments
chunk_size = 1e6

#orc_indices = np.arange(0,10)
orc_indices = [3,4]

if __name__ == '__main__':
    with Pool(100) as p:
        p.map(do_ind, orc_indices)
