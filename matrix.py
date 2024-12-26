import pandas as pd
from scipy.sparse import csr_matrix


def construct_sparse_user_item_matrix(transactions_df):
    """
    Constructs a sparse user-item matrix from a transactions DataFrame.

    Parameters:
        transactions_df (pd.DataFrame): A DataFrame containing at least two columns: 'user_id' and 'item_name'.

    Returns:
        csr_matrix: A sparse matrix where rows represent users, columns represent items,
                    and values represent the number of times a user bought an item.
        user_mapping (dict): A dictionary mapping user IDs to row indices in the matrix.
        item_mapping (dict): A dictionary mapping item names to column indices in the matrix.
    """
    # Ensure the input DataFrame has the required columns
    if 'user_id' not in transactions_df.columns or 'item_name' not in transactions_df.columns:
        raise ValueError("The input DataFrame must contain 'user_id' and 'item_name' columns.")

    # Create mappings for users and items to unique indices
    user_mapping = {user_id: idx for idx, user_id in enumerate(transactions_df['user_id'].unique())}
    item_mapping = {item_name: idx for idx, item_name in enumerate(transactions_df['item_name'].unique())}

    # Map users and items to their respective indices
    transactions_df['user_idx'] = transactions_df['user_id'].map(user_mapping)
    transactions_df['item_idx'] = transactions_df['item_name'].map(item_mapping)

    # Create a sparse matrix using scipy's csr_matrix
    row_indices = transactions_df['user_idx']
    col_indices = transactions_df['item_idx']
    data = [1] * len(transactions_df)  # Each transaction counts as 1 initially

    # Aggregate counts for duplicate (user, item) pairs
    sparse_matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(user_mapping), len(item_mapping))
    )

    # Sum duplicate entries to count purchases
    sparse_matrix.sum_duplicates()

    return sparse_matrix, user_mapping, item_mapping