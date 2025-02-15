import numpy as np
import os
import pyttb
import pandas as pd
from functools import partial
from multiprocessing import Pool, Manager
from scipy.sparse import csr_matrix, save_npz, load_npz
import tqdm
import pyarrow.parquet as pq
import gc


def compute_b_from_g(G, λ=100):
    diagIndices = np.diag_indices(G.shape[0])
    G[diagIndices] += λ
    P = np.linalg.inv(G)
    B = P / (-np.diag(P))
    B[diagIndices] = 0
    return B


def apply_dict(arr, my_dict):
    vectorized_get = np.vectorize(my_dict.get)
    return vectorized_get(arr)


if __name__ == '__main__':

    #with Manager() as manager:
    res_dir = '75M results'
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

    # load sparse human-cat interaction matrix
    x_mat_filename = os.path.join(res_dir, f'user cat matrix {postfix}.npz')
    X = load_npz(x_mat_filename)
    print(X.shape)

    # Initialize the pool with the custom initializer
    print('Computing...')

    ###############################################
    chunk_size = 10**6
    parts = nhumans//chunk_size + 1 # we split large UMC 3D tensor into smaller parts

    humans_list = umc_subs[:, 0]
    for i, hinds in tqdm.tqdm(enumerate(np.array_split(np.arange(nhumans), parts)), total=parts):
        # rel_... means something relevant to this chunk (part of a big array or smth)
        # ch_... means actual inside-chunk indices or data

        print(f'chunk {i}')
        lb, rb = min(hinds), max(hinds) # borders of the chunk
        print('borders', lb, rb)

        # inds of global UMC subs relevant to this chunk
        rel_inds = np.where((humans_list >= lb) & (humans_list < rb))[0]
        rel_umc_subs = umc_subs[rel_inds,:]

        # renumber humans for this particular chunk
        rel_unique_humans = np.unique(rel_umc_subs[:,0])
        rel_humans_to_ch_humans_mapping = dict(zip(rel_unique_humans, range(len(rel_unique_humans))))
        ch_humans = apply_dict(rel_umc_subs[:,0], rel_humans_to_ch_humans_mapping)

        # inds of non-empty human-month pairs relevant to this chunk
        rel_nonempty_inds = np.where((h_nonempty_inds >= lb) & (h_nonempty_inds < rb))[0]
        rel_hm_pair_humans = h_nonempty_inds[rel_nonempty_inds]
        ch_hm_pair_humans = apply_dict(rel_hm_pair_humans, rel_humans_to_ch_humans_mapping)
        # since we are not slicing on months, rel_hm_pair_months and ch_hm_pair_months are the same
        ch_hm_pair_months = m_nonempty_inds[rel_nonempty_inds]

        # slice of UMC tensor containing data for this chunk
        A = np.zeros((len(rel_unique_humans), nmonths, ncat), dtype=bool)
        # again, rel_ and ch_ are the same for months and categories
        A[ch_humans, rel_umc_subs[:,1], rel_umc_subs[:,2]] = True

        # targets
        targets = A[ch_hm_pair_humans, ch_hm_pair_months, :]

        # previous month - leave False where not applicable
        prev_ms1 = ch_hm_pair_months - 1
        has_preceding = np.where(prev_ms1 >= 0)[0]
        prev1 = np.zeros((len(ch_hm_pair_humans), ncat), dtype=bool)

        # fill data from previous months where applicable
        ch_hm_pair_humans_has_preceding = ch_hm_pair_humans[has_preceding]
        ch_hm_pair_months_has_preceding = ch_hm_pair_months[has_preceding]
        prev1[ch_hm_pair_humans_has_preceding, :] = \
            A[ch_hm_pair_humans_has_preceding, ch_hm_pair_months_has_preceding - 1, :]

        # collaborative human feats - recomputing from scratch to save RAM
        # using relative inds, because X contains all data and is not chunked
        collab_feats = np.array(X[rel_hm_pair_humans, :].todense().dot(B))

        # Initialize an empty dictionary to store feature data
        data_dict = {}

        print(targets.shape)
        print(prev1.shape)
        print(collab_feats.shape)

        # Add prev1 features to the dictionary
        for j, bool_array in enumerate(prev1.T):
            data_dict[f"prev1_{j+1}"] = bool_array

        # Add himan features to the dictionary
        for j, float_array in enumerate(collab_feats.T):
            data_dict[f"collab_{j+1}"] = float_array

        # Add target features to the dictionary
        for j, bool_array in enumerate(targets.T):
            data_dict[f"target_{j+1}"] = bool_array

        # add month as a feature
        data_dict['month'] = ch_hm_pair_months//12 + 1

        #print(data_dict)
        # Create a DataFrame from the dictionary
        df = pd.DataFrame(data_dict)

        print(df.head())
        os.makedirs(os.path.join(res_dir, 'dataset'), exist_ok=True)
        df.to_parquet(os.path.join(res_dir, 'dataset', f'df ch{i} {postfix}.parquet'))
        del df
        gc.collect()
