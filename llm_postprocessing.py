from preprocessing import preprocess_group, get_all_preprocessed_subcategories
from navec import Navec
import numpy as np
import pandas as pd

npath = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(npath)

def process_string(s):
    sp = s.lower().split('-')[0].replace(':', '').replace(',', '').replace("'", '').replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('.', '')
    return sp if len(sp)!=0 else None

def get_av_vector(item, enable_stemming=False):
    #print('item:', item)
    parts = item.split(' ')
    #print('parts:', parts)
    meaning = [process_string(p) for p in parts]
    #print('meaning', meaning)
    print('navec', [w for w in meaning if w in navec])
    vecs = [navec.get(w) for w in parts if w in navec]
    if len(vecs) == 0:
        avvec = None
    else:
        avvec = np.sum(vecs, axis=0)

    return avvec


llm_items = pd.read_excel('LLM items.xlsx')
for i, row in llm_items.iterrows():
    #print(row)
    sc = row['subcategory']
    print(sc)
    items_to_process = row['10 products'].split(',')[:10]
    #print(len(items_to_process))
    if len(items_to_process) == 9 and 'и' in items_to_process[-1]:
        last_2_items = items_to_process[-1].split(' и ')
        final_items_to_process = items_to_process[:-1] + last_2_items + [sc]
    elif len(items_to_process) == 10:
        final_items_to_process = items_to_process + [sc]
    else:
        final_items_to_process = items_to_process + [sc for _ in range(11 - len(items_to_process))]


    print(len(final_items_to_process), final_items_to_process)
    vectors = [get_av_vector(item) for item in final_items_to_process]
    sc_avvec = get_av_vector(sc)
    print('=======================')