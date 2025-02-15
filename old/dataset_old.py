import numpy as np
import os
import pyttb
import pandas as pd
from functools import partial
from multiprocessing import Pool, Manager
from scipy.sparse import csr_matrix, save_npz, load_npz
import tqdm
import pyarrow.parquet as pq


def compute_b_from_g(G, λ=100):
    diagIndices = np.diag_indices(G.shape[0])
    G[diagIndices] += λ
    P = np.linalg.inv(G)
    B = P / (-np.diag(P))
    B[diagIndices] = 0
    return B

'''
def process_nonzero_pair(i,
                         X=None,
                         B=None,
                         h_inds=None,
                         m_inds=None,
                         umc_subs=None,
                         ncat=None):
'''
def process_nonzero_pair(i):
    h = h_inds[i]
    m = m_inds[i]

    # prev month
    prev_m1 = m - 1
    if prev_m1 < 0:
        cat_data_prev1 = np.zeros(ncat, dtype=bool)
    else:
        cat_indices_prev1 = np.where((umc_subs[:, 0] == h) & (umc_subs[:, 1] == prev_m1))[0]
        cat_data_prev1 = np.zeros(ncat, dtype=bool)
        cat_data_prev1[cat_indices_prev1] = True

    # target
    cat_indices = np.where((umc_subs[:, 0] == h) & (umc_subs[:, 1] == m))[0]
    positive_cats = umc_subs[cat_indices, 2]
    cat_data = np.zeros(ncat, dtype=bool)
    cat_data[positive_cats] = True

    # human features
    #hfeats = human_feats[h,:]
    hfeats = X[h,:].todense().dot(B)

    return hfeats, cat_data_prev1, cat_data


def xray_slicing(arr, vals_to_search1, vals_to_search2):
    '''
    returns values of the third columns based on multiple searches in first and second
    :param arr:
    :param vals_to_search1:
    :param vals_to_search2:
    :return:
    '''
    # Vectorized search: Create a boolean mask for each condition
    k = len(vals_to_search1)

    mask1 = arr[:, 0][:, None] == vals_to_search1[None, :]  # Match values in the first column
    mask2 = arr[:, 1][:, None] == vals_to_search2[None, :]  # Match values in the second column

    # Combine masks using logical AND to find rows matching both conditions
    combined_mask = mask1 & mask2

    # Extract indices where combined_mask is True
    row_indices = np.where(combined_mask)[0]

    # Extract corresponding values from the third column
    result = arr[row_indices, 2]

    # split into parts corresponding to each search
    print('starting conversion')
    counts = np.sum(combined_mask.astype(int), axis=0)
    borders = np.zeros(k+1)
    borders[1:] = np.cumsum(counts)
    borders = borders.astype(int)

    res_parts = [result[borders[i]: borders[i + 1]] for i in range(k)]
    return res_parts


# Initialization function to set up the global variable in each worker
def init_worker(x, b, hi, mi, subs, nc):
    global X
    X = x

    global B
    B = x

    global h_inds
    h_inds = hi

    global m_inds
    m_inds = mi

    global umc_subs
    umc_subs = subs

    global ncat
    ncat = nc


NJOBS = 2

if __name__ == '__main__':

    #with Manager() as manager:
    res_dir = '../75M results'
    postfix = '75M'

    # load user-mouth-cat tensor
    umc_filename = os.path.join(res_dir, f"sparse_user_month_cat_{postfix}.npz")
    umc_ = np.load(umc_filename)
    umc_subs = umc_['subs']
    umc_vals = umc_['vals']
    all_nnz = len(umc_subs) # total number of user-month-cat interactions

    umc = pyttb.sptensor.from_aggregator(umc_subs, umc_vals)  # Sparse tensor users-months-cats
    print(umc.shape)
    nhumans, nmonths, ncat = umc.shape


    # collapse over categories
    um_sums = umc.collapse(dims=np.array([2])).double()
    print(um_sums.shape)
    h_nonempty_inds, m_nonempty_inds = np.where(um_sums != 0)  # nonempty user-month pairs
    nnz_hm = len(h_nonempty_inds) # number of nonempty user-month pairs
    print('nonzero pairs:', nnz_hm)

    # load G and compute B
    g_mat_filename = os.path.join(res_dir, f'G matrix {postfix}.npz')
    G = load_npz(g_mat_filename).todense()
    B = compute_b_from_g(G)

    # load sparse user-cat interaction matrix
    x_mat_filename = os.path.join(res_dir, f'user cat matrix {postfix}.npz')
    X = load_npz(x_mat_filename)
    print(X.shape)

    # Initialize the pool with the custom initializer
    print('Computing...')
    '''
    with Pool(NJOBS,
              initializer=init_worker,
              initargs=(X, B, h_nonempty_inds, m_nonempty_inds, umc_subs, ncat)) as pool:
    '''
    '''
    all_res = pool.map(process_nonzero_pair,
                                 np.arange(1000))
    '''

    '''
    all_res = tqdm.tqdm(pool.map(partial(process_nonzero_pair,
                               X=X,
                               B=B,
                               h_inds=h_nonempty_inds,
                               m_inds=m_nonempty_inds,
                               umc_subs=umc_subs,
                               ncat=ncat),
                       np.arange(1000)))
    '''

    ###############################################
    chunk_size = 10**7
    parts = all_nnz//chunk_size + 1 # we split large UMC 3D tensor into smaller parts

    humans_list = umc_subs[:, 0]
    for inds in tqdm.tqdm(np.array_split(np.arange(all_nnz), parts)):
        lb, rb = min(inds), max(inds) # borders of the chunk
        ch_inds = np.where(humans_list < )[0]
        #print(ch_inds)
        ch_umc_subs = umc_subs[ch_inds,:]
        ch_nonempty_inds = np.where(h_nonempty_inds < ch)[0]
        ch_hm_pair_humans = h_nonempty_inds[ch_nonempty_inds]
        ch_hm_pair_months = m_nonempty_inds[ch_nonempty_inds]

    print(len(ch_inds), len(ch_nonempty_inds))

    A = np.zeros((ch, nmonths, ncat), dtype=bool)
    A[ch_umc_subs[:,0], ch_umc_subs[:,1], ch_umc_subs[:,2]] = True
    print(A.shape)
    print(np.count_nonzero(A))

    V = A[ch_hm_pair_humans, ch_hm_pair_months, :]
    print(V.shape)
    print(np.count_nonzero(V))
    #comb_nonempty_flag = h_nonempty_flag & m_nonempty_flag
    #print(np.sum(comb_nonempty_flag.astype(int)))

    #cats_ = umc[h_nonempty_inds, m_nonempty_inds]
    #print(cats_)
    '''
    for c in range(1):
        all_hfeats = []
        all_cat_data_prev1 = []
        all_cat_data = []

        for inds in tqdm.tqdm(np.array_split(np.arange(chunk_size), parts)):
            #for i in tqdm.tqdm(np.arange(10)):
            hs = h_nonempty_inds[inds]
            ms = m_nonempty_inds[inds]

            # prev month
            cat_data_prev1 = np.zeros((psize, ncat), dtype=bool)
            prev_ms1 = ms-1
            has_preceding = np.where(prev_ms1 >= 0)[0]

            # calculate for months from 1-st only, otherwise zeros are left
            positive_cats_prev1 = xray_slicing(umc_subs, hs[has_preceding], prev_ms1[has_preceding])
            for i, positive_cats_prev1_part in enumerate(positive_cats_prev1):
                cat_data_prev1[i, positive_cats_prev1_part] = True

            # target
            cat_data = np.zeros((psize, ncat), dtype=bool)
            positive_cats = xray_slicing(umc_subs, hs, ms)
            for i, positive_cats_part in enumerate(positive_cats):
                cat_data[i, positive_cats_part] = True

            # human features
            # hfeats = human_feats[h,:]
            hfeats = np.array(X[hs, :].todense().dot(B))

            all_hfeats.append(hfeats)
            all_cat_data_prev1.append(cat_data_prev1)
            all_cat_data.append(cat_data)

        # Initialize an empty dictionary to store feature data
        data_dict = {}

        targets = np.vstack(all_cat_data).T
        prev1 = np.vstack(all_cat_data_prev1).T
        collab_feats = np.vstack(all_hfeats).T

        print(targets.shape)
        print(prev1.shape)
        print(collab_feats.shape)

        # Add prev1 features to the dictionary
        for i, bool_array in enumerate(prev1):
            data_dict[f"prev1_{i+1}"] = bool_array

        # Add himan features to the dictionary
        for i, float_array in enumerate(collab_feats):
            data_dict[f"cf_{i+1}"] = float_array

        # Add target features to the dictionary
        for i, bool_array in enumerate(targets):
            data_dict[f"t_{i+1}"] = bool_array

        # Create a DataFrame from the dictionary
        df = pd.DataFrame(data_dict)

        print(df.head())
        df.to_parquet(os.path.join(res_dir, f'df c{c} {postfix}.parquet'))
        '''