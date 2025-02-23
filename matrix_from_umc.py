import os
import numpy as np
import pyttb
from scipy.sparse import csr_matrix, save_npz
import gc

res_dir = '10 orcs'
postfix = '10 orcs'

umc_filename = os.path.join(res_dir, f"sparse_user_month_cat_{postfix}.npz")
umc_ = np.load(umc_filename)
umc_subs = umc_['subs']
umc_vals = umc_['vals']
print('initial indices shape:')
print(umc_subs.shape)

months_to_exclude =[max(umc_subs[:,1]), max(umc_subs[:,1])-1]
inds_to_exclude = np.isin(umc_subs[:,1], months_to_exclude)

filt_umc_subs = umc_subs[~inds_to_exclude]
filt_umc_vals = umc_vals[~inds_to_exclude]

del umc_subs, umc_vals
gc.collect()

print('filtered indices shape:')
print(filt_umc_subs.shape)
all_nnz = len(filt_umc_subs)  # total number of user-month-cat interactions

umc = pyttb.sptensor.from_aggregator(filt_umc_subs, filt_umc_vals)  # Sparse tensor users-months-cats
print('tensor shape:', umc.shape)
nhumans, nmonths, ncat = umc.shape

del filt_umc_subs, filt_umc_vals
gc.collect()

# collapse over month
uc_sums = umc.collapse(dims=np.array([1]))
print(uc_sums.subs.shape)

del umc
gc.collect()

print('Constructing sparse matrix...')
# Aggregate counts for duplicate (user, item) pairs
X = csr_matrix(
    (uc_sums.vals.ravel(), (uc_sums.subs[:,0], uc_sums.subs[:,1])),
    shape=(nhumans, ncat)
)

# Sum duplicate entries to count transactions
X.sum_duplicates()

del uc_sums
gc.collect()

print('Saving sparse matrix...')
mat_filename = os.path.join(res_dir, f'user cat matrix {postfix}.npz')
save_npz(mat_filename, X)

