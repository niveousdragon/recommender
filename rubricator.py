from navec import Navec

from preprocessing import preprocess_group, get_all_preprocessed_subcategories
from llm_postprocessing import process_string
import pandas as pd
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
import tqdm
from multiprocessing import Pool, Process
from functools import partial


def get_product_avvector(data:tuple, EMBS=None):
    data = [process_string(s) for s in data]
    vecs = []
    for d in data:
        if d is None or len(d)==0:
            continue

        vec = EMBS.get(d)
        vecs.append(vec)

    arrvecs = np.array([v for v in vecs if v is not None])
    if len(arrvecs) == 0:
        return np.ones(300)
    else:
        return np.mean(arrvecs, axis=0)


def apply_rubricator_to_dataframe(df_name,
                                  df_path=None,
                                  navec=None,
                                  emb_norm=None,
                                  sc_mapping=None,
                                  spath=None,
                                  orc_ind=None):

    df = pd.read_csv(os.path.join(df_path, df_name))
    # preprocess_products
    all_products = []
    for p in tqdm.tqdm(df['product']):
        if p != '[]':
            prepr_product = preprocess_group(p)
            all_products.append(prepr_product)
        else:
            all_products.append(None)

    # get all product embeddings (dummy ones possible)
    all_prod_avvecs = [get_product_avvector(p, EMBS=navec) if p is not None else np.ones(300) for p in all_products]

    # get indices of non-dummy product embeddings
    prod_emb_sums = np.array([np.sum(emb) for emb in all_prod_avvecs])
    good_prod_indices = np.where(prod_emb_sums != 300.0)[0]

    # calculate tensor of cosine similarities
    prod_avvec_matrix = np.vstack(all_prod_avvecs).T[:, good_prod_indices].astype(np.float32)
    # prod_avvec_matrix_norm = prod_avvec_matrix / norm(prod_avvec_matrix, axis=0, keepdims=True)
    prod_avvec_matrix_norm = prod_avvec_matrix / np.sqrt(np.sum(prod_avvec_matrix ** 2, axis=0, keepdims=True)).astype(
        np.float32)
    # print(prod_avvec_matrix_norm.shape)

    M_cos = np.tensordot(emb_norm, prod_avvec_matrix_norm, axes=([2], [0]))
    # print(np.mean(M, axis=1).shape)
    #--------print(M_cos.shape)

    M_max = np.max(M_cos, axis=1)
    #--------print('max', M_max.shape)

    mask1 = M_max > 0.8
    mask2 = M_max > 0.96
    #--------print('masks', mask2.shape)

    # compute final score
    S = np.mean(M_cos, axis=1) + np.multiply(M_max, mask1) + 100 * np.multiply(M_max, mask2)
    # S = np.apply_along_axis(score_from_cossims, axis=1, arr=M_cos)
    # S = np.mean(M_cos, axis=1) + np.max(M_high, axis=1) + 100 * np.max(M_coincidence, axis=1)
    #--------print(S.shape)

    topcat_inds = np.argmax(S, axis=0)
    topcat_sims = S[topcat_inds, np.arange(len(topcat_inds))]
    topcat = [sc_mapping[tci] for tci in topcat_inds]

    light_df = pd.DataFrame()
    light_df['human'] = df['human_id'][good_prod_indices]
    light_df['date'] = df['date_actual'][good_prod_indices]
    light_df['product'] = df['product'][good_prod_indices]
    light_df['top_cat'] = topcat
    light_df['top_cat_sim'] = topcat_sims

    save = 1
    if save:
        df_name_base = df_name[:-4]
        # print(df_name_base)
        print(f'saving results for {df_name_base}...')

        save_path = os.path.join(spath, 'full-scale v2 rubricator result', f'file {orc_ind}')
        os.makedirs(save_path, exist_ok=True)

        min_thr = 0.2
        light_df = light_df[light_df['top_cat_sim'] > min_thr]
        light_df.to_excel(os.path.join(save_path,
                              f'{df_name_base} neuro cat light.xlsx'),
                             engine='xlsxwriter',
                             index=False)

    return True


if __name__ == '__main__':
    npath = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
    navec = Navec.load(npath)

    PATH = "C:\\Projects\\temp"
    ORC_INDS = [2]
    NJOBS = 5

    emb_arr = np.load('embeddings arr v1.2.npz')['arr_0'].transpose(2, 0, 1)
    emb_norm = emb_arr / norm(emb_arr, axis=2, keepdims=True).astype(np.float32)

    print(emb_arr.shape)
    llm_items = pd.read_excel('LLM items corr full v1.2.xlsx')
    sc_mapping = dict(llm_items['subcategory'])

    for orc_ind in ORC_INDS:
        DFPATH = os.path.join(PATH, 'full-scale v1', f'file {orc_ind}')
        all_dfs = os.listdir(DFPATH)

        with Pool(NJOBS) as pool:
            _ = pool.map(partial(apply_rubricator_to_dataframe,
                                 df_path=DFPATH,
                                 emb_norm=emb_norm,
                                 navec=navec,
                                 sc_mapping=sc_mapping,
                                 spath=PATH,
                                 orc_ind=orc_ind),
                                 all_dfs[:])



