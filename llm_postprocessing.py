from preprocessing import preprocess_group, get_all_preprocessed_subcategories
from navec import Navec
import numpy as np
import pandas as pd
import ast

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
    #print('navec', [w for w in meaning if w in navec])
    vecs = [navec.get(w) for w in meaning if w in navec]
    if len(vecs) == 0:
        avvec = np.zeros(300)
    else:
        avvec = np.sum(vecs, axis=0)

    return avvec


llm_items = pd.read_excel('LLM items corr full v1.0.xlsx')

all_embeddings = []
for i, row in llm_items.iterrows():
    #print(row)
    sc_ = row['subcategory']
    sc = " ".join(ast.literal_eval(sc_))
    items_to_process = row['10 products'][1:-1].split(',')[:10]
    #print(len(items_to_process))
    if len(items_to_process) == 9:
        if ' и ' in items_to_process[-1]:
            last_2_items = items_to_process[-1].split(' и ')
            final_items_to_process = items_to_process[:-1] + last_2_items + [sc]
        else:
            final_items_to_process = items_to_process + [sc for _ in range(11 - len(items_to_process))]

    elif len(items_to_process) == 10:
        final_items_to_process = items_to_process + [sc]

    else:
        final_items_to_process = items_to_process + [sc for _ in range(11 - len(items_to_process))]

    vectors = [get_av_vector(item) for item in final_items_to_process]
    sc_avvec = get_av_vector(sc)

    corr_vectors = np.array([v if not np.allclose(v, np.zeros(300), rtol=1e-5) else sc_avvec for v in vectors])
    if len(corr_vectors) != 11:
        print(sc)
        print(len(items_to_process))
        print(len(final_items_to_process), final_items_to_process)
        print(corr_vectors.shape)
        print('=======================')
    all_embeddings.append(corr_vectors)

all_embs_array = np.dstack(all_embeddings)
np.savez('embeddings arr v1.0.npz', all_embs_array)