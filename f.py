import pandas as pd
import numpy as np
from tqdm import tqdm  # For progress bar
import os
import multiprocessing as mp
from scipy.sparse import coo_matrix
import gc

# ---------------------- Configuration ----------------------

# Paths to your CSV files
DATABASE_PATH = '5_alteast4_movielens_data.csv'
AUX_DATA_PATH = '5_percent_mldataset_intersected_adversary_data_for.csv'
WEIGHTS_PATH = '5_movie_weights.csv'

# Output directories for results
OUTPUT_DIR_UNIFORM = 'deanonymization_results_uniform'
OUTPUT_DIR_NO_TOP_100 = 'deanonymization_results_no_top_100'
OUTPUT_DIR_NO_TOP_500 = 'deanonymization_results_no_top_500'

# Ensure output directories exist
os.makedirs(OUTPUT_DIR_UNIFORM, exist_ok=True)
os.makedirs(OUTPUT_DIR_NO_TOP_100, exist_ok=True)
os.makedirs(OUTPUT_DIR_NO_TOP_500, exist_ok=True)

# Batch size for processing ml users
BATCH_SIZE = 1000

# ---------------------- Load Datasets ----------------------

print("Loading datasets...")
database = pd.read_csv(DATABASE_PATH)
aux_data = pd.read_csv(AUX_DATA_PATH)
weights_df = pd.read_csv(WEIGHTS_PATH)

# Get unique 'imdbId' values from both dataframes
unique_aux_ids = set(aux_data['imdbId'].unique())
unique_ml_ids = set(database['imdbId'].unique())

# Check if the difference is zero
if unique_aux_ids - unique_ml_ids == set():
    print("All GOOD, no extra imdbId found in auxiliary data.")
else:
    print("Warning: Auxiliary data contains imdbIds not present in MovieLens data.")

# Drop rows with empty 'rating_value' in aux_data and reset index
aux_data = aux_data.dropna(subset=['rating_value']).reset_index(drop=True)

# Half the ratings as MovieLens scale is 0-5
aux_data['rating_value'] = aux_data['rating_value'] / 2

# ---------------------- Preprocessing ----------------------

# Convert 'imdbId' to strings to ensure consistency
database['imdbId'] = database['imdbId'].astype(str)
aux_data['imdbId'] = aux_data['imdbId'].astype(str)

# Rename 'userId' columns to avoid confusion
database = database.rename(columns={'userId': 'ml_userId'})
aux_data = aux_data.rename(columns={'userId': 'aux_userId'})

# Convert weights to a dictionary for fast lookup
movie_weights = dict(zip(weights_df['imdbId'].astype(str), weights_df['weight']))

# ---------------------- Create imdbId to aux_userIds Mapping ----------------------

print("Creating imdbId to aux_userIds mapping...")
# Group aux_data by 'imdbId' and aggregate 'aux_userId' into sets for fast lookup
imdb_to_aux_users = aux_data.groupby('imdbId')['aux_userId'].apply(set).to_dict()
print(f"Total unique movies in auxiliary data: {len(imdb_to_aux_users)}")

# ---------------------- Get Unique ml_userIds ----------------------

ml_user_ids = database['ml_userId'].unique()
total_ml_users = len(ml_user_ids)
print(f"Total unique MovieLens users: {total_ml_users}")

# ---------------------- Create Unified imdbId Mapping ----------------------

# Step 1: Create a Unified Set of imdbId Values
all_imdbIds = pd.unique(pd.concat([database['imdbId'], aux_data['imdbId']]))
all_imdbIds = sorted(all_imdbIds)  # Ensure consistent ordering
num_movies = len(all_imdbIds)
print(f"Total unique movies across both datasets: {num_movies}")

# Step 2: Map imdbId to Column Indices
imdbId_to_col_idx = {imdbId: idx for idx, imdbId in enumerate(all_imdbIds)}

# Step 3a: Map ML Users to Row Indices
ml_userIds_sorted = sorted(database['ml_userId'].unique())
ml_userId_to_row_idx = {userId: idx for idx, userId in enumerate(ml_userIds_sorted)}
num_ml_users = len(ml_userIds_sorted)
print(f"Total unique MovieLens users: {num_ml_users}")

# Step 3b: Map Auxiliary Users to Row Indices
aux_userIds_sorted = sorted(aux_data['aux_userId'].unique())
aux_userId_to_row_idx = {userId: idx for idx, userId in enumerate(aux_userIds_sorted)}
num_aux_users = len(aux_userIds_sorted)
print(f"Total unique auxiliary users: {num_aux_users}")

# ---------------------- Build Sparse Matrices ----------------------

from scipy.sparse import coo_matrix

print("Building sparse matrices...")

# Prepare data for the MovieLens ratings and timestamps matrices
database_rows = []
database_cols = []
database_ratings_data = []
database_timestamps_data = []

for idx, row in tqdm(database.iterrows(), total=database.shape[0], desc='Processing MovieLens ratings'):
    userId = row['ml_userId']
    imdbId = row['imdbId']
    rating = row['rating']
    timestamp = row['timestamp']
    
    row_idx = ml_userId_to_row_idx[userId]
    col_idx = imdbId_to_col_idx[imdbId]
    
    database_rows.append(row_idx)
    database_cols.append(col_idx)
    database_ratings_data.append(rating)
    database_timestamps_data.append(timestamp)

database_ratings_matrix = coo_matrix((database_ratings_data, (database_rows, database_cols)),
                                     shape=(num_ml_users, num_movies)).tocsr()

database_timestamps_matrix = coo_matrix((database_timestamps_data, (database_rows, database_cols)),
                                        shape=(num_ml_users, num_movies)).tocsr()

# Prepare data for the auxiliary ratings and timestamps matrices
aux_rows = []
aux_cols = []
aux_ratings_data = []
aux_timestamps_data = []

for idx, row in tqdm(aux_data.iterrows(), total=aux_data.shape[0], desc='Processing auxiliary ratings'):
    userId = row['aux_userId']
    imdbId = row['imdbId']
    rating = row['rating_value']
    timestamp = row['review_date_epoch']
    
    if pd.isna(rating):
        continue  # Skip entries without valid ratings
    
    row_idx = aux_userId_to_row_idx[userId]
    col_idx = imdbId_to_col_idx.get(imdbId)
    if col_idx is None:
        continue  # Should not happen if we've combined all imdbIds
    
    aux_rows.append(row_idx)
    aux_cols.append(col_idx)
    aux_ratings_data.append(rating)
    aux_timestamps_data.append(timestamp)

aux_ratings_matrix = coo_matrix((aux_ratings_data, (aux_rows, aux_cols)),
                                shape=(num_aux_users, num_movies)).tocsr()

aux_timestamps_matrix = coo_matrix((aux_timestamps_data, (aux_rows, aux_cols)),
                                   shape=(num_aux_users, num_movies)).tocsr()

# ---------------------- Prepare Weights Array ----------------------

print("Preparing weights array...")
weights_df['imdbId'] = weights_df['imdbId'].astype(str)
weights_df['col_idx'] = weights_df['imdbId'].map(imdbId_to_col_idx)
weights_df = weights_df.dropna(subset=['col_idx'])
weights = np.zeros(num_movies)

weights[weights_df['col_idx'].astype(int)] = weights_df['weight'].values

# ---------------------- Similarity Functions ----------------------

def compute_rating_similarity(ml_ratings, candidate_ratings, rating_threshold=1):
    """
    Compute rating similarity based on threshold.
    """
    # Compute absolute difference
    rating_diff = np.abs(ml_ratings - candidate_ratings)
    # Valid ratings are those where both users have rated the movie
    valid_mask = (ml_ratings > 0) & (candidate_ratings > 0)
    # Ratings are similar if the difference is within the threshold
    rating_sim = np.zeros_like(ml_ratings)
    rating_sim[valid_mask] = (rating_diff[valid_mask] <= rating_threshold).astype(float)
    return rating_sim

def compute_timestamp_similarity(ml_timestamps, candidate_timestamps, time_threshold=14 * 24 * 3600):
    """
    Compute timestamp similarity based on threshold.
    """
    # Compute absolute difference in timestamps
    time_diff = np.abs(ml_timestamps - candidate_timestamps)
    # Valid timestamps are those where both users have timestamps
    valid_mask = (ml_timestamps > 0) & (candidate_timestamps > 0)
    # Timestamps are similar if the difference is within the threshold
    timestamp_sim = np.zeros_like(ml_timestamps)
    timestamp_sim[valid_mask] = (time_diff[valid_mask] <= time_threshold).astype(float)
    return timestamp_sim

def compute_rating_presence_similarity(ml_ratings, candidate_ratings):
    """
    Compute rating similarity based on presence only.
    """
    # Create a mask where both users have rated the movie (presence check)
    valid_mask = (ml_ratings > 0) & (candidate_ratings > 0)
    # Set similarity to 1.0 if both users have rated the movie, 0.0 otherwise
    rating_sim = np.zeros_like(ml_ratings)
    rating_sim[valid_mask] = 1.0
    return rating_sim

def compute_timestamp_presence_similarity(ml_timestamps, candidate_timestamps):
    """
    Compute timestamp similarity based on presence only.
    """
    # Create a mask where both users have timestamps (presence check)
    valid_mask = (ml_timestamps > 0) & (candidate_timestamps > 0)
    # Set similarity to 1.0 if both users have timestamps, 0.0 otherwise
    timestamp_sim = np.zeros_like(ml_timestamps)
    timestamp_sim[valid_mask] = 1.0
    return timestamp_sim

# ---------------------- Initialize Global Variables for Workers ----------------------

# These global variables will be shared across worker processes
global_ml_userId_to_row_idx = None
global_aux_userId_to_row_idx = None
global_database_ratings_matrix = None
global_database_timestamps_matrix = None
global_aux_ratings_matrix = None
global_aux_timestamps_matrix = None
global_weights = None
global_ml_userIds_sorted = None
global_aux_userIds_sorted = None

def initializer(ml_userId_to_row, aux_userId_to_row, db_ratings, db_timestamps, aux_ratings, aux_timestamps, weights_array, ml_users_sorted, aux_users_sorted):
    """
    Initializer function for worker processes to set global variables.
    """
    global global_ml_userId_to_row_idx
    global global_aux_userId_to_row_idx
    global global_database_ratings_matrix
    global global_database_timestamps_matrix
    global global_aux_ratings_matrix
    global global_aux_timestamps_matrix
    global global_weights
    global global_ml_userIds_sorted
    global global_aux_userIds_sorted
    
    global_ml_userId_to_row_idx = ml_userId_to_row
    global_aux_userId_to_row_idx = aux_userId_to_row
    global_database_ratings_matrix = db_ratings
    global_database_timestamps_matrix = db_timestamps
    global_aux_ratings_matrix = aux_ratings
    global_aux_timestamps_matrix = aux_timestamps
    global_weights = weights_array
    global_ml_userIds_sorted = ml_users_sorted
    global_aux_userIds_sorted = aux_users_sorted

# ---------------------- Worker Function ----------------------

def worker_compute_scores(args):
    """
    Worker function to compute scores for a single ML user.
    
    Args:
    - ml_user_id (int): The MovieLens user ID to de-anonymize.
    - candidate_aux_user_ids (list of int): List of auxiliary user IDs who have at least one movie rating in common.
    - phi (float): Eccentricity threshold to determine a unique match.
    
    Returns:
    - dict or None: Returns a dictionary with match info if a unique match is found, else None.
    """
    ml_user_id, candidate_aux_user_ids, phi = args
    
    # Get ml_user_idx
    ml_user_idx = global_ml_userId_to_row_idx.get(ml_user_id)
    if ml_user_idx is None:
        return None  # ML user ID not found

    # Get the indices of the movies rated by the ml_user_id
    ml_rated_movies = global_database_ratings_matrix.getrow(ml_user_idx).indices
    ml_ratings = global_database_ratings_matrix.getrow(ml_user_idx).data
    ml_timestamps = global_database_timestamps_matrix.getrow(ml_user_idx).data
    ml_weights = global_weights[ml_rated_movies]

    num_movies = len(ml_rated_movies)

    if num_movies == 0:
        # ml_user_id has not rated any movies
        return None

    # Map candidate_aux_user_ids to row indices, ensuring they exist
    candidate_aux_user_indices = [
        global_aux_userId_to_row_idx.get(aux_user_id)
        for aux_user_id in candidate_aux_user_ids
    ]
    # Remove None values (aux_user_ids not found)
    candidate_aux_user_indices = [idx for idx in candidate_aux_user_indices if idx is not None]

    num_candidates = len(candidate_aux_user_indices)

    if num_candidates == 0:
        return None  # No candidates found

    # Initialize an array to accumulate scores
    scores = np.zeros(num_candidates)

    # Process auxiliary users in sub-batches to manage memory usage
    sub_batch_size = 1000  # Adjust based on available memory
    for start in range(0, num_candidates, sub_batch_size):
        end = min(start + sub_batch_size, num_candidates)
        sub_batch_indices = candidate_aux_user_indices[start:end]

        # Get candidate ratings and timestamps matrices (sub_batch_size x num_movies)
        candidate_ratings = global_aux_ratings_matrix[sub_batch_indices][:, ml_rated_movies].toarray()
        candidate_timestamps = global_aux_timestamps_matrix[sub_batch_indices][:, ml_rated_movies].toarray()

        # Broadcast ml_ratings and ml_timestamps to match sub-batch size
        # Shape: (sub_batch_size, num_movies)
        ml_ratings_matrix = np.tile(ml_ratings, (len(sub_batch_indices), 1))
        ml_timestamps_matrix = np.tile(ml_timestamps, (len(sub_batch_indices), 1))

        # Compute similarities
        rating_sim = compute_rating_similarity(ml_ratings_matrix, candidate_ratings)
        timestamp_sim = compute_timestamp_similarity(ml_timestamps_matrix, candidate_timestamps)

        sim = rating_sim * timestamp_sim

        # Multiply similarities by weights and sum over movies
        # Each movie's weight is applied to the corresponding column
        # Broadcasting ml_weights across rows
        weighted_sim = sim * ml_weights

        # Sum across movies to get total score for each auxiliary user in the sub-batch
        sub_batch_scores = np.sum(weighted_sim, axis=1)

        # Accumulate the scores into the main scores array
        scores[start:end] += sub_batch_scores

        # Free memory for this sub-batch
        del candidate_ratings, candidate_timestamps, ml_ratings_matrix, ml_timestamps_matrix, rating_sim, timestamp_sim, sim, weighted_sim, sub_batch_scores
        gc.collect()

    # After processing all sub-batches, determine if there's a unique match
    max_score_idx = np.argmax(scores)
    max_score = scores[max_score_idx]
    sorted_scores = np.sort(scores)[::-1]
    max2_score = sorted_scores[1] if len(sorted_scores) > 1 else 0
    sigma = np.std(scores)
    eccentricity = (max_score - max2_score) / sigma if sigma > 0 else np.inf

    min_score_threshold = 0.1  # Define a minimum score required for a valid match

    if eccentricity < phi or max_score < min_score_threshold:
        return None  # No unique match found
    else:
        # Return the matched auxiliary user and score
        matched_aux_user_idx = candidate_aux_user_indices[max_score_idx]
        matched_aux_userId = global_aux_userIds_sorted[matched_aux_user_idx]
        return {'ml_user_id': ml_user_id, 'aux_user_id': matched_aux_userId, 'score': max_score}

# ---------------------- Parallel Processing Setup ----------------------

def main():
    phi = 1.5  # Eccentricity threshold
    matches = []

    # List of batch files
    batch_files = [os.path.join(OUTPUT_DIR_NO_TOP_500, f) for f in os.listdir(OUTPUT_DIR_NO_TOP_500) if f.endswith('.csv')]

    # Prepare arguments for the worker function
    tasks = []

    for batch_file in batch_files:
        # Load the batch_results_df
        batch_results_df = pd.read_csv(batch_file)

        for idx, row in batch_results_df.iterrows():
            ml_user_id = row['ml_user_id']
            aux_users_str = row['aux_users']
            if not aux_users_str:
                continue  # No candidates

            # Convert aux_users_str to a list of candidate IDs
            candidate_aux_user_ids = [int(aux_user_id) for aux_user_id in aux_users_str.split(',') if aux_user_id]

            # Append the task
            tasks.append((ml_user_id, candidate_aux_user_ids, phi))

        # Free memory for the DataFrame after processing each batch file
        del batch_results_df
        gc.collect()

    print(f"Total tasks to process: {len(tasks)}")

    # Initialize the multiprocessing pool with an initializer
    pool = mp.Pool(processes=16, initializer=initializer,
                   initargs=(ml_userId_to_row_idx, aux_userId_to_row_idx,
                             database_ratings_matrix, database_timestamps_matrix,
                             aux_ratings_matrix, aux_timestamps_matrix,
                             weights, ml_userIds_sorted, aux_userIds_sorted))

    # Process tasks in parallel
    for result in tqdm(pool.imap_unordered(worker_compute_scores, tasks), total=len(tasks), desc='Processing ML users'):
        if result is not None:
            matches.append(result)
            # Optionally, save matches incrementally to disk to manage memory
            if len(matches) % 10000 == 0:
                temp_df = pd.DataFrame(matches)
                if not os.path.exists('deanonymization_matches_temp.csv'):
                    temp_df.to_csv('deanonymization_matches_temp.csv', index=False)
                else:
                    temp_df.to_csv('deanonymization_matches_temp.csv', mode='a', header=False, index=False)
                matches = []
                del temp_df
                gc.collect()

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    # Save any remaining matches
    if matches:
        matches_df = pd.DataFrame(matches)
        matches_df.to_csv('deanonymization_matches.csv', index=False)
    else:
        print("No matches to save.")

    print(f"Total matches found: {len(matches)}")

if __name__ == '__main__':
    main()
