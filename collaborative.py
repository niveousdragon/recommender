import numpy as np
import os
from scipy.sparse import csr_matrix, save_npz, load_npz
import matplotlib.pyplot as plt

compute_gram_matrix = False
compute_features = True

res_dir = '75M results'
postfix = '75M'

mat_filename = os.path.join(res_dir, f'user cat matrix {postfix}.npz')
g_mat_filename = os.path.join(res_dir, f'G matrix {postfix}.npz')

if compute_gram_matrix:
    X = load_npz(mat_filename)
    G = X.T.dot(X)
    save_npz(g_mat_filename, G)
else:
    G = load_npz(g_mat_filename).todense()

print(G.shape)
λ = 100
diagIndices = np.diag_indices(G.shape[0])
G[diagIndices] += λ
P = np.linalg.inv(G)
B = P / (-np.diag(P))
B[diagIndices] = 0

b_mat_filename = os.path.join(res_dir, f'B matrix {postfix}.npz')
save_npz(b_mat_filename, B)
'''
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(B)
plt.show()
'''

feat_mat_filename = os.path.join(res_dir, f'feat matrix {postfix} lambda={λ}.npz')
if compute_features:
    X = load_npz(mat_filename)
    F = X.todense().dot(B)
    np.savez(feat_mat_filename, F)