
import pandas as pd
import numpy as np
from scipy.stats import entropy
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # For progress bar

# ---------------------- Configuration ----------------------

# Paths to your CSV files
DATABASE_PATH = '5_alteast4_movielens_data.csv'
AUX_DATA_PATH = '5_percent_mldataset_intersected_adversary_data_for.csv'
WEIGHTS_PATH = '5_movie_weights.csv'

# Load datasets
print("Loading datasets...")
database = pd.read_csv(DATABASE_PATH)
aux_data = pd.read_csv(AUX_DATA_PATH)
weights_df = pd.read_csv(WEIGHTS_PATH)

# Convert weights to a dictionary for fast lookup
# Assumes 'imdbId' is the common identifier in both datasets
movie_weights = dict(zip(weights_df['imdbId'], weights_df['weight']))


def filter_top_n_movies(database, top_n):
    """
    Exclude the top N most popular movies from the database.
    Popularity is determined by the number of ratings (frequency).
    """
    if top_n <= 0:
        return database
    top_movies = database['imdbId'].value_counts().head(top_n).index
    filtered_db = database[~database['imdbId'].isin(top_movies)]
    return filtered_db

# Group database by 'userId'
database_grouped = database.groupby('userId')

# Group auxiliary data by 'reviewer_name'
aux_data_grouped = aux_data.groupby('reviewer_name')

# Function to calculate user similarity using the vectorized approach
def calculate_user_similarity_vectorized(aux_profile, db_profile, rating_thresh, date_thresh, weight_func):
    """
    Calculate the similarity score between a target user and a database user using vectorized NumPy operations.
    """
    # Find common movies
    common_movies = list(set(aux_profile.keys()).intersection(set(db_profile.keys())))
    if not common_movies:
        return 0  # No similarity if no common movies

    # Extract data using vectorized operations
    aux_ratings = np.array([aux_profile[movie]['rating_value'] for movie in common_movies])
    db_ratings = np.array([db_profile[movie]['rating'] for movie in common_movies])
    aux_dates = np.array([aux_profile[movie]['review_date_epoch'] for movie in common_movies])
    db_dates = np.array([db_profile[movie]['timestamp'] for movie in common_movies])
    weights = np.array([weight_func(movie) for movie in common_movies])

    # Calculate differences
    rating_diffs = np.abs(aux_ratings / 2 - db_ratings)  # Adjust scaling if necessary
    date_diffs = np.abs(aux_dates - db_dates)

    # Ensure thresholds are positive and non-zero
    rating_thresh = max(1e-6, rating_thresh)
    date_thresh = max(1e-6, date_thresh)

    # Calculate similarity components using vectorized operations
    rating_sims = np.exp(-(rating_diffs / rating_thresh))
    date_sims = np.exp(-(date_diffs / date_thresh))

    # Aggregate the similarity scores using weights
    total_score = np.sum(weights * (rating_sims + date_sims))

    return total_score

# Function to aggregate auxiliary user profiles
def aggregate_auxiliary_profiles(aux_group):
    """
    Aggregate the auxiliary data into user profiles.
    Each auxiliary user profile is a dictionary mapping imdbId to a dictionary with rating_value and review_date_epoch.
    """
    aux_users = {}
    for name, group in aux_group:
        aux_users[name] = group.set_index('movie_title')[['rating_value', 'review_date_epoch']].to_dict('index')
    return aux_users

# Function to aggregate database user profiles
def aggregate_user_profiles(db_group):
    """
    Aggregate the database data into user profiles.
    Each database user profile is a dictionary mapping imdbId to a dictionary with rating and timestamp.
    """
    db_users = {}
    for user_id, group in db_group:
        db_users[user_id] = group.set_index('imdbId')[['rating', 'timestamp']].to_dict('index')
    return db_users



def de_anonymization_best_guess_vectorized(scores):
    """
    Perform 'Best-guess' de-anonymization using vectorized NumPy operations.
    """
    if scores.size == 0:
        return None, None

    max_score = np.max(scores)
    if scores.size > 1:
        max2_score = np.partition(scores, -2)[-2]
    else:
        max2_score = 0
    sigma = np.std(scores)

    # Compute eccentricity in a vectorized manner
    eccentricity = (max_score - max2_score) / sigma if sigma > 0 else float('inf')
    if eccentricity >= 1.5:
        return max_score, eccentricity
    else:
        return None, eccentricity

def de_anonymization_best_guess_vectorized(scores):
    """
    Perform 'Best-guess' de-anonymization using vectorized NumPy operations.
    """
    if scores.size == 0:
        return None, None

    max_score = np.max(scores)
    if scores.size > 1:
        max2_score = np.partition(scores, -2)[-2]
    else:
        max2_score = 0
    sigma = np.std(scores)

    # Compute eccentricity in a vectorized manner
    eccentricity = (max_score - max2_score) / sigma if sigma > 0 else float('inf')
    if eccentricity >= 1.5:
        return max_score, eccentricity
    else:
        return None, eccentricity


# Aggregate user profiles
print("Aggregating user profiles...")
db_users = aggregate_user_profiles(database_grouped)
aux_users = aggregate_auxiliary_profiles(aux_data_grouped)

# Function to run user-to-user similarity calculations
def run_user_similarity_comparison(aux_users, db_users, rating_thresh, date_thresh, weight_func):
    results = []
    for aux_user, aux_profile in tqdm(aux_users.items(), desc="Comparing auxiliary users"):
        scores = []
        for db_user, db_profile in db_users.items():
            score = calculate_user_similarity_vectorized(aux_profile, db_profile, rating_thresh, date_thresh, weight_func)
            scores.append((db_user, score))
        # Sort scores by descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        # Store the top result (best match) for each auxiliary user
        results.append({
            'aux_user': aux_user,
            'best_match_user': scores[0][0],
            'max_score': scores[0][1] if scores else 0
        })
    return results

# Configuration parameters
rating_thresh = 1.0  # Adjust as needed
date_thresh = 14 * 24 * 3600  # 14 days in seconds

# Run the comparison
print("Running user-to-user similarity comparison...")
results = run_user_similarity_comparison(aux_users, db_users, rating_thresh, date_thresh, lambda movie: movie_weights.get(movie, 1))

# Display results
print("\n=== Results ===")
for result in results:
    print(f"Auxiliary User: {result['aux_user']}, Best Match User: {result['best_match_user']}, Max Score: {result['max_score']:.4f}")
