import numpy as np
import os
from scipy.sparse import csr_matrix, save_npz, load_npz
import matplotlib.pyplot as plt

compute_gram_matrix = True
compute_features = True

#res_dir = '75M results'
res_dir = 'user_item_results'

#postfix = '75M'
postfix = '0-5_orcs'


mat_filename = os.path.join(res_dir, f'user cat matrix {postfix}.npz')
g_mat_filename = os.path.join(res_dir, f'G matrix {postfix}.npz')

if compute_gram_matrix:
    X = load_npz(mat_filename)
    G = X.T.dot(X)
    save_npz(g_mat_filename, G)
else:
    G = load_npz(g_mat_filename)

print(G.shape)
G = G.todense()

λ = 100
diagIndices = np.diag_indices(G.shape[0])
G[diagIndices] += λ
print(type(G))
P = np.linalg.inv(G)
B = P / (-np.diag(P))
B[diagIndices] = 0

b_mat_filename = os.path.join(res_dir, f'B matrix {postfix} lambda={λ}.npz')
print(type(B))

np.savez(b_mat_filename, B)

'''
fig, ax = plt.subplots(figsize=(10,10))
ax.matshow(B)
plt.show()
'''

if compute_features:

    feat_mat_filename = os.path.join(res_dir, f'feat matrix {postfix} lambda={λ}.npz')

    X = load_npz(mat_filename)
    F = X.todense().dot(B)
    np.savez(feat_mat_filename, F)
