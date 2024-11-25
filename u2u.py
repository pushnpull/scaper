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




# Configuration for experiments
# Each experiment has:
# - exp_id: unique identifier
# - rating_thresh: threshold for rating similarity
# - date_thresh: threshold for date similarity (in seconds)
# - type: 'Best-guess' or 'Entropic'
# - exclude_top_n_movies: exclude top N popular movies from the database
experiments_config = [
    {"exp_id": 4, "rating_thresh": 0, "date_thresh": 3*24*3600, "type": "Best-guess", "exclude_top_n_movies": 0},
    {"exp_id": 5, "rating_thresh": 0, "date_thresh": 14*24*3600, "type": "Best-guess", "exclude_top_n_movies": 0},
    {"exp_id": 6, "rating_thresh": 0, "date_thresh": 14*24*3600, "type": "Entropic", "exclude_top_n_movies": 0},
    {"exp_id": 8, "rating_thresh": 0, "date_thresh": float('inf'), "type": "Best-guess", "exclude_top_n_movies": 100},
    {"exp_id": 9, "rating_thresh": 1, "date_thresh": 14*24*3600, "type": "Best-guess", "exclude_top_n_movies": 0},
    {"exp_id": 10, "rating_thresh": 1, "date_thresh": 14*24*3600, "type": "Best-guess", "exclude_top_n_movies": 0},
    {"exp_id": 11, "rating_thresh": 0, "date_thresh": float('inf'), "type": "Entropic", "exclude_top_n_movies": 500},
    {"exp_id": 12, "rating_thresh": 1, "date_thresh": 14*24*3600, "type": "Best-guess", "exclude_top_n_movies": 0}
]

# ---------------------- Helper Functions ----------------------

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

def aggregate_user_profiles(database):
    """
    Aggregate the database into user profiles.
    Each user profile is a dictionary mapping imdbId to {'rating': ..., 'timestamp': ...}
    """
    print("Aggregating user profiles...")
    database_users = database.groupby('userId').apply(
        lambda x: x.set_index('imdbId')[['rating', 'timestamp']].to_dict('index')
    ).to_dict()
    return database_users

def aggregate_auxiliary_profiles(aux_data):
    """
    Aggregate the auxiliary data into user profiles.
    Each auxiliary user profile is a dictionary mapping imdbId to {'rating_value': ..., 'review_date_epoch': ...}
    """
    print("Aggregating auxiliary user profiles...")
    # Check if 'imdbId' exists; if not, attempt to map from 'movie_title'
    if 'imdbId' in aux_data.columns:
        aux_users = aux_data.groupby('userId').apply(
            lambda x: x.set_index('imdbId')[['rating_value', 'review_date_epoch']].to_dict('index')
        ).to_dict()
    elif 'movie_title' in aux_data.columns:
        # Attempt to map 'movie_title' to 'imdbId' using weights_df
        if 'movie_title' in weights_df.columns:
            print("Mapping 'movie_title' to 'imdbId'...")
            title_to_id = dict(zip(weights_df['movie_title'], weights_df['imdbId']))
            aux_data['imdbId'] = aux_data['movie_title'].map(title_to_id)
            # Drop records where 'imdbId' could not be mapped
            aux_data = aux_data.dropna(subset=['imdbId'])
            # Convert 'imdbId' to integer
            aux_data['imdbId'] = aux_data['imdbId'].astype(int)
            aux_users = aux_data.groupby('userId').apply(
                lambda x: x.set_index('imdbId')[['rating_value', 'review_date_epoch']].to_dict('index')
            ).to_dict()
        else:
            raise ValueError("Cannot map 'movie_title' to 'imdbId'; 'weights_df' lacks 'movie_title'")
    else:
        # If no 'userId', treat all aux_data as a single target user
        # Assuming 'imdbId' is available
        if 'imdbId' in aux_data.columns:
            aux_users = {
                'target_user': aux_data.set_index('imdbId')[['rating_value', 'review_date_epoch']].to_dict('index')
            }
        else:
            raise ValueError("aux_data must have either 'imdbId' or 'movie_title' column")
    return aux_users

def calculate_user_similarity(aux_profile, db_profile, rating_thresh, date_thresh, weight_func):
    """
    Calculate the similarity score between a target user and a database user.
    """
    # Find common movies
    common_movies = set(aux_profile.keys()).intersection(set(db_profile.keys()))
    if not common_movies:
        return 0  # No similarity if no common movies

    total_score = 0
    for movie in common_movies:
        aux_rating = aux_profile[movie]['rating_value']
        db_rating = db_profile[movie]['rating']
        aux_date = aux_profile[movie]['review_date_epoch']
        db_date = db_profile[movie]['timestamp']

        # Calculate differences
        rating_diff = abs(aux_rating / 2 - db_rating)  # Adjust scaling if necessary
        date_diff = abs(aux_date - db_date)

        # Ensure thresholds are positive and non-zero
        rating_thresh = max(1e-6, rating_thresh)
        date_thresh = max(1e-6, date_thresh)

        # Calculate similarity components
        rating_sim = np.exp(-(rating_diff / rating_thresh))
        date_sim = np.exp(-(date_diff / date_thresh))

        # Get weight based on movie popularity
        weight = weight_func(movie)

        # Aggregate the similarity scores
        total_score += weight * (rating_sim + date_sim)

    return total_score

def de_anonymization_best_guess(scores):
    """
    Perform 'Best-guess' de-anonymization by selecting the highest scoring user
    and checking if the top score is significantly higher than the second.
    """
    if len(scores) == 0:
        return None, None

    max_score = np.max(scores)
    if len(scores) > 1:
        # Find the second highest score
        max2_score = np.partition(scores, -2)[-2]
    else:
        max2_score = 0
    sigma = np.std(scores)
    # Compute eccentricity
    eccentricity = (max_score - max2_score) / sigma if sigma > 0 else float('inf')
    if eccentricity >= 1.5:
        return max_score, eccentricity
    else:
        return None, eccentricity

def de_anonymization_entropic(scores):
    """
    Perform 'Entropic' de-anonymization by calculating the entropy of the probability distribution
    over candidate users based on similarity scores.
    """
    if len(scores) == 0:
        return None, float('inf')

    sigma = np.std(scores) if np.std(scores) > 0 else 1e-6
    probabilities = np.exp(scores / sigma)
    probabilities /= probabilities.sum()
    shannon_entropy = entropy(probabilities, base=2)
    return probabilities, shannon_entropy

def run_experiment(exp_config, filtered_database_users, aux_users):
    """
    Run a single experiment based on the provided configuration.
    """
    results = []
    exp_id = exp_config["exp_id"]
    rating_thresh = exp_config["rating_thresh"]
    date_thresh = exp_config["date_thresh"]
    de_type = exp_config["type"]

    # Iterate over each target user in the auxiliary sample
    for target_user_id, aux_profile in tqdm(aux_users.items(), desc=f"Experiment {exp_id}"):
        # Compute similarity scores with all users in the filtered database
        user_scores = {}
        for db_user_id, db_profile in filtered_database_users.items():
            score = calculate_user_similarity(aux_profile, db_profile, rating_thresh, date_thresh, movie_weights.get)
            user_scores[db_user_id] = score

        # Convert scores to numpy array for processing
        scores = np.array(list(user_scores.values()))

        if de_type == "Best-guess":
            max_score, eccentricity = de_anonymization_best_guess(scores)
            # Find the user(s) with the max score
            if max_score is not None:
                best_match_user_ids = [user_id for user_id, score in user_scores.items() if score == max_score]
                # Assuming unique best match
                best_match_user_id = best_match_user_ids[0] if best_match_user_ids else None
                results.append({
                    "exp_id": exp_id,
                    "target_user": target_user_id,
                    "best_match_user": best_match_user_id,
                    "max_score": max_score,
                    "eccentricity": eccentricity,
                    "status": "Success" if best_match_user_id else "Failed"
                })
            else:
                results.append({
                    "exp_id": exp_id,
                    "target_user": target_user_id,
                    "best_match_user": None,
                    "max_score": max_score,
                    "eccentricity": eccentricity,
                    "status": "Ambiguous"
                })
        elif de_type == "Entropic":
            probabilities, shannon_entropy = de_anonymization_entropic(scores)
            # Depending on the experiment needs, you can analyze or store the entropy
            # For example, to determine if the entropy is low enough to imply a unique match
            # For now, store the entropy and probabilities
            results.append({
                "exp_id": exp_id,
                "target_user": target_user_id,
                "probabilities": probabilities.tolist(),
                "shannon_entropy": shannon_entropy,
                "status": "Calculated"
            })
        else:
            raise ValueError(f"Unknown de_type: {de_type}")

    return results

def run_all_experiments_parallel(experiments_config, prefiltered_databases, aux_users):
    """
    Run all experiments in parallel using multiprocessing.
    """
    # Prepare a list of (config, filtered_database_users, aux_users) tuples
    args = []
    for config in experiments_config:
        exclude_n = config["exclude_top_n_movies"]
        filtered_database_users = prefiltered_databases.get(exclude_n)
        args.append((config, filtered_database_users, aux_users))

    def run_experiment_wrapper(args_tuple):
        config, filtered_database_users, a_users = args_tuple
        return run_experiment(config, filtered_database_users, a_users)

    # Use multiprocessing Pool to run experiments in parallel
    with Pool(cpu_count()) as pool:
        # Use imap with a progress bar
        all_results = list(tqdm(pool.imap(run_experiment_wrapper, args), total=len(args), desc="Running all experiments"))

    return all_results

# ---------------------- Main Execution ----------------------

if __name__ == "__main__":
    # Aggregate auxiliary user profiles
    print("Aggregating auxiliary user profiles...")
    aux_users = aggregate_auxiliary_profiles(aux_data)

    # Identify unique 'exclude_top_n_movies' values
    unique_excludes = set([config["exclude_top_n_movies"] for config in experiments_config])

    # Precompute filtered_database_users for each unique exclude_n
    prefiltered_databases = {}
    for exclude_n in unique_excludes:
        if exclude_n == 0:
            print(f"\nAggregating database user profiles without excluding any movies (exclude_top_n_movies={exclude_n})...")
            filtered_database = database.copy()
        else:
            print(f"\nFiltering out top {exclude_n} movies by popularity...")
            filtered_database = filter_top_n_movies(database, exclude_n)
        print(f"Aggregating user profiles for exclude_top_n_movies={exclude_n}...")
        filtered_database_users = aggregate_user_profiles(filtered_database)
        prefiltered_databases[exclude_n] = filtered_database_users

    # Run all experiments in parallel
    print("\nRunning experiments in parallel...")
    results = run_all_experiments_parallel(experiments_config, prefiltered_databases, aux_users)

    # Process and print results
    for exp_idx, exp_results in enumerate(results):
        exp_config = experiments_config[exp_idx]
        exp_id = exp_config["exp_id"]
        print(f"\n=== Results for Experiment {exp_id} ===")
        for res in exp_results:
            if res["type"] == "Best-guess":
                print(f"Target User: {res['target_user']}, Best Match: {res['best_match_user']}, "
                      f"Max Score: {res['max_score']:.4f}, Eccentricity: {res['eccentricity']:.4f}, "
                      f"Status: {res['status']}")
            elif res["type"] == "Entropic":
                print(f"Target User: {res['target_user']}, Shannon Entropy: {res['shannon_entropy']:.4f}, "
                      f"Status: {res['status']}")
            else:
                print(res)  # For any unforeseen types
