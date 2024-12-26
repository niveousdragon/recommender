from navec import Navec

from processing import PATH
from preprocessing import preprocess_group, get_all_preprocessed_subcategories
import pandas as pd
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
import tqdm
from multiprocessing import Pool, Process

npath = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(npath)

dfname = 'test fd 3'
df = pd.read_csv(os.path.join(PATH, 'test', dfname + '.csv'))

def cossim(a, b):
    if a is None or b is None:
        return None
    cos = dot(a, b) / (norm(a) * norm(b))
    return float(cos)

all_subcats, reverse_cat = get_all_preprocessed_subcategories(enable_stemming=False)


def get_emb_vectors(data):
    # build single and average vectors for subcategories
    all_avvecs = {}
    all_vecs = {}
    for d in tqdm.tqdm(data):
        if d is None:
            continue

        vecs = [navec.get(w) for w in d if w in navec]
        if len(vecs) == 0:
            all_vecs.update({d: [None]})
            #all_avvecs.update({d: [None]})
        else:
            all_vecs.update({d: vecs})
            #avvec = np.sum(vecs, axis=0)
            #all_avvecs.update({d: avvec})

    return all_vecs, all_avvecs


all_sc_vecs, all_sc_avvecs = get_emb_vectors(all_subcats)


# preprocess_products
all_products = []
for p in tqdm.tqdm(df['product']):
    if p != '[]':
        prepr_product = preprocess_group(p)
        all_products.append(prepr_product)
    else:
        all_products.append(None)


all_prod_vecs, all_prod_avvecs = get_emb_vectors(all_products)


def get_cat_prod_sim_matrix(embs_cat, embs_prod):
    M = np.zeros((len(embs_cat), len(embs_prod)))
    for i, ec in enumerate(embs_cat):
        for j, ep in enumerate(embs_prod):
            cos = cossim(ec, ep)
            if cos is not None:
                M[i,j] = cos
            else:
                M[i,j] = 0
    return M


def process_cat_prod_sim_matrix(M, sim_mode=0):
    if sim_mode == 0: # mean of max for each category word
        return np.mean(np.max(M, axis=1))
    if sim_mode == 1: # mean of max for each category word
        return np.mean(M)


def get_product_subcat(p, sim_mode=0):
    if p is not None:
        if len(p)!=0:
            try:
                pemb = all_prod_vecs[p]
                sc_scores = {}
                for sc in all_sc_vecs.keys():
                    cemb = all_sc_vecs[sc]
                    M = get_cat_prod_sim_matrix(cemb, pemb)
                    cp_score = process_cat_prod_sim_matrix(M, sim_mode=sim_mode)
                    sc_scores.update({sc: cp_score})

                #sims = np.array([cossim(pemb, all_sc_avvecs.get(sc)) if sc in all_sc_avvecs.keys() else -1 for sc in all_subcats])
                sorted_scores = sorted(sc_scores.items(), key=lambda item: item[1])
                top = sorted_scores[-3:]

                if len(top) != 0:
                    '''
                    print('product:', p)
                    print('probable categories:')
                    for tcat, tscore in top[::-1]:
                        print(tcat, np.round(tscore, 3))
                    print('------------')
                    '''
                    return [(tc, np.round(ts, 3)) for tc, ts in top[::-1]]
                else:
                    return None
            except KeyError:
                return None
        else:
            return None
    else:
        return None


def is_row_valid(row):
    try:
        row.astype(str).apply(lambda x: x.encode('utf-8'))
        return True
    except Exception:
        return False


def clean_df(df):
    valid_rows = df.apply(is_row_valid, axis=1)
    clean_df = df[valid_rows]
    clean_df.dropna(inplace=True)
    return clean_df


if __name__ == '__main__':
    '''
    pool = [Process(target=get_product_subcat, kwargs={'sim_mode':0}]
    with pool:
        topres = pool.map(get_product_subcat, all_products)
    '''

    with Pool(32) as pool:
        topres = pool.map(get_product_subcat, all_products)

    print('saving...')
    light_df = pd.DataFrame()
    light_df['human'] = df['human_id']
    light_df['product'] = df['product']
    light_df['top_cat'] = [tr[0][0] if tr is not None else None for tr in topres]
    light_df['top2_cat'] = [tr[1][0] if tr is not None else None for tr in topres]
    light_df['top3_cat'] = [tr[2][0] if tr is not None else None for tr in topres]
    light_df['top_cat_sim'] = [tr[0][1] if tr is not None else None for tr in topres]
    light_df['top2_cat_sim'] = [tr[1][1] if tr is not None else None for tr in topres]
    light_df['top3_cat_sim'] = [tr[2][1] if tr is not None else None for tr in topres]
    #light_df['top_cat2'] = [tr[0][0] if tr is not None else None for tr in topres2]
    #light_df['top_cat_sim2'] = [tr[0][1] if tr is not None else None for tr in topres2]

    df['top_cat'] = topres
    #df['top_cat2'] = topres2
    print(df.head())
    save = 1
    if save:
        clean_df(df).to_excel(os.path.join(PATH, 'test', f'{dfname} cat.xlsx'),
                              engine='xlsxwriter',
                              index=False)

        clean_df(light_df).to_excel(os.path.join(PATH, 'test', f'{dfname} cat light.xlsx'),
                                    engine = 'xlsxwriter',
                                    index=False)