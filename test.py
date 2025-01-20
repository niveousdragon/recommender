from navec import Navec
import numpy as np
from numpy.linalg import norm
import os
from processing import PATH
import json
import pandas as pd
from llm_postprocessing import npath
from rubricator import cossim, get_product_avvector

npath = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(npath)

#print(navec.sim('пирожок', "автокресло"))

emb_arr = np.load('embeddings arr v1.1.npz')['arr_0'].transpose(2,0,1)
emb_norm = norm(emb_arr, axis=-1)

'''
v0 = navec.get('пирожок')
print(v0[:20])

v1 = get_product_avvector(('пирожок','булочка','автомобиль'), EMBS=navec)
print(v1[:20])
print([cossim(emb_arr[35,i], v0) for i in range(11)])
print([cossim(emb_arr[35,i], v1) for i in range(11)])
'''
#print(emb_arr[35,2][:20])
#print(navec.get('автокресло')[:20])

with open('total counts.json') as json_file:
    cnt = json.load(json_file)

df = pd.Series(cnt).astype(float)
sorted_series = df.sort_values(ascending=False)
print(sorted_series)

#sorted_series.to_excel('total counts 20M.xlsx')