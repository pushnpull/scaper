import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.pyplot as plt

# ---------------------- Configuration ----------------------

# Paths to your CSV files
DATABASE_PATH = '5_alteast4_movielens_data.csv'

# ---------------------- Load Dataset ----------------------

print("Loading dataset...")
database = pd.read_csv(DATABASE_PATH)

# ---------------------- Preprocessing ----------------------

# Convert 'imdbId' to strings to ensure consistency
database['imdbId'] = database['imdbId'].astype(str)

# Rename 'userId' columns to avoid confusion
database = database.rename(columns={'userId': 'ml_userId'})

# ---------------------- Filter Top N Movies ----------------------

def filter_top_n_movies(database, n=500):
    """
    Filters out the top N most rated movies from the MovieLens dataset.

    Parameters:
    - database (DataFrame): MovieLens dataset.
    - n (int): Number of top movies to exclude.

    Returns:
    - filtered_database (DataFrame)
    """
    print(f"Excluding top {n} most popular movies...")

    # Identify top N movies based on MovieLens dataset
    top_movies = database['imdbId'].value_counts().head(n).index.tolist()

    # Filter out top N movies from the dataset
    filtered_database = database[~database['imdbId'].isin(top_movies)].reset_index(drop=True)

    print(f"Database size after filtering: {filtered_database.shape}")

    return filtered_database

# ---------------------- Apply Filtering ----------------------

# Choose which movies to exclude
EXCLUDE_TOP_N = 500
database = filter_top_n_movies(database, n=EXCLUDE_TOP_N)

# ---------------------- Prepare Sparse Matrix ----------------------

unique_ml_ids = set(database['imdbId'].unique())
ml_imdbIds = sorted(unique_ml_ids)
num_movies = len(ml_imdbIds)
print(f"Total unique movies: {num_movies}")

# Step 2: Map imdbId to Column Indices
imdbId_to_col_idx = {imdbId: idx for idx, imdbId in enumerate(ml_imdbIds)}

# Step 3: Map Users to Row Indices
ml_userIds = sorted(database['ml_userId'].unique())
ml_userId_to_row_idx = {userId: idx for idx, userId in enumerate(ml_userIds)}
num_ml_users = len(ml_userIds)
print(f"Total unique MovieLens users: {num_ml_users}")

# Prepare data for the ratings matrix
database_rows = []
database_cols = []
database_ratings_data = []

for idx, row in tqdm(database.iterrows(), total=database.shape[0], desc='Processing database ratings'):
    userId = row['ml_userId']
    imdbId = row['imdbId']

    row_idx = ml_userId_to_row_idx[userId]
    col_idx = imdbId_to_col_idx[imdbId]

    database_rows.append(row_idx)
    database_cols.append(col_idx)
    # We only care about presence, so we can set data to 1
    database_ratings_data.append(1)

# Create the presence matrix (users x movies)
presence_matrix = coo_matrix((database_ratings_data, (database_rows, database_cols)),
                             shape=(num_ml_users, num_movies)).tocsr()

# ---------------------- Compute Maximum Similarities ----------------------

print("Computing maximum similarities for each user...")

# Precompute the length of supports for all users
user_support_lengths = np.diff(presence_matrix.indptr)

max_similarities = np.zeros(num_ml_users)

# Iterate over each user
for i in tqdm(range(num_ml_users), desc='Computing similarities'):
    presence_vector_i = presence_matrix.getrow(i)  # 1 x num_movies sparse vector

    # Compute the numerator vector (intersection sizes with other users)
    numerator_vector = presence_vector_i.dot(presence_matrix.T)  # 1 x num_users sparse vector

    # Convert numerator_vector to COO format to access indices and data
    numerator_vector = numerator_vector.tocoo()

    # Get the indices (user indices) and data (intersection sizes)
    indices = numerator_vector.col
    data = numerator_vector.data

    # Exclude self-similarity explicitly by removing the entry where index equals 'i'
    mask = indices != i
    candidate_users = indices[mask]
    numerators = data[mask]

    # Calculate Jaccard similarity
    len_supp_i = user_support_lengths[i]
    len_supp_j = user_support_lengths[candidate_users]

    denominators = len_supp_i + len_supp_j - numerators

    similarities = numerators / denominators

    # Store the maximum similarity excluding self-comparison
    if similarities.size > 0:
        max_similarity = similarities.max()
        max_similarities[i] = max_similarity
    else:
        max_similarities[i] = 0.0  # No other users with common movies

# ---------------------- Plot the Distribution ----------------------

print("Plotting the distribution of maximum similarities...")

# Sort the max similarities in ascending order
sorted_similarities = np.sort(max_similarities)

# Compute the cumulative distribution
cumulative = np.arange(1, num_ml_users + 1) / num_ml_users

# Plot the complementary cumulative distribution function (CCDF)
plt.figure(figsize=(8, 6))
plt.plot(sorted_similarities, 1 - cumulative, drawstyle='steps-post')
plt.xlabel('Similarity to the nearest neighbor (x)')
plt.ylabel('Fraction of users with nearest neighbor similarity ≥ x')
plt.title('Sparsity of the MovieLens Dataset (Excluding Self-Comparison)')
plt.grid(True)
plt.show()

# Optionally, save the plot
plt.savefig('movielens_sparsity_plot.png')
