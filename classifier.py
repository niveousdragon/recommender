import pyarrow.parquet as pq
import pyarrow as pa
import concurrent.futures
import glob
import os
from catboost import Pool, CatBoostClassifier
from sklearn.multiclass import OneVsRestClassifier


def load_parquet_chunk(file_path, target=None):
    data = pq.read_table(file_path).to_pandas()
    features = [f for f in data.columns if 't_' not in f]

    if target is None:
        target = [f for f in data.columns if 't_' in f]
    else:
        target = [f't_{target}']

    return data[features], data[target]


res_dir = '75M results'
postfix = '75M'
i=0
datapath = os.path.join(res_dir, 'dataset', f'df ch{i} {postfix}.parquet')
X, y = load_parquet_chunk(datapath, target=None)

print(y)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    cat_features=['month'],
    loss_function='Logloss',
    task_type='CPU',  # Use 'CPU' if GPU is not available
    thread_count=-1,
    #devices='0',  # Specify GPU device if using GPU
    verbose=100
)

#model.fit(X,y)


# Wrap it with OneVsRestClassifier for multilabel classification
ovr_classifier = OneVsRestClassifier(model, n_jobs=-1)

# Fit the model
ovr_classifier.fit(X, y.to_numpy().astype(int))

# Make predictions
predictions = ovr_classifier.predict(X)
# 5. Save the final model
model.save_model('catboost_model.cbm')

'''
# 6. Make predictions (example)
X_test, y_test = load_data_chunk('test_data.csv')
test_pool = Pool(X_test)
predictions = model.predict(test_pool)
'''
