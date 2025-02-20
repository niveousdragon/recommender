import pyarrow.parquet as pq
import pyarrow as pa
import concurrent.futures
import glob
import os
import numpy as np
from catboost import Pool, CatBoostClassifier
from sklearn.multiclass import OneVsRestClassifier


def load_parquet_chunk(file_path, targets=None):
    data = pq.read_table(file_path).to_pandas()
    features = [f for f in data.columns if 't_' not in f]

    if targets is None:
        tnames = [f for f in data.columns if f in 't_' in f]
    else:
        tnames = [f't_{k + 1}' for k in targets]

    targets = [f for f in data.columns if f in tnames]
    return data[features], data[targets]


res_dir = '75M results'
postfix = '75M'
i=0
datapath = os.path.join(res_dir, 'dataset', f'df ch{i} {postfix}.parquet')

data = pq.read_table(datapath).to_pandas()
features = [f for f in data.columns if 't_' not in f]
X = data[features]

for j in range(10):
    targets = [f't_{j+1}']
    y = data[targets]

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.2,
        depth=10,
        cat_features=['month'],
        loss_function='Logloss',
        task_type='GPU',  # Use 'CPU' if GPU is not available
        #thread_count=-1,
        devices='0',  # Specify GPU device if using GPU
        verbose=True,
        #use_best_model=True
    )

    model.fit(X,y)
    # 5. Save the final model
    model.save_model(f'catboost_model_ch{i}_t{j}.cbm')

'''
# Wrap it with OneVsRestClassifier for multilabel classification
ovr_classifier = OneVsRestClassifier(model, n_jobs=-1)

# Fit the model
ovr_classifier.fit(X, y.to_numpy().astype(int))

# Make predictions
predictions = ovr_classifier.predict(X)

'''
'''
# 6. Make predictions (example)
X_test, y_test = load_data_chunk('test_data.csv')
test_pool = Pool(X_test)
predictions = model.predict(test_pool)
'''
