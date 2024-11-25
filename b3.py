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

# Eccentricity parameter
PHI = 1.5

# Parameters for scoring function
RHO0 = 1.5  # Rating threshold parameter
D0 = 30 * 24 * 60 * 60  # Date threshold in seconds (30 days)

# ---------------------- Load Datasets ----------------------

print("Loading datasets...")
database = pd.read_csv(DATABASE_PATH)
aux_data = pd.read_csv(AUX_DATA_PATH)
weights_df = pd.read_csv(WEIGHTS_PATH)

# ---------------------- Preprocessing ----------------------

# Convert 'imdbId' to ensure consistency
database['imdbId'] = database['imdbId'].astype(str).str.lstrip('0')
aux_data['imdbId'] = aux_data['imdbId'].astype(str).str.lstrip('0')

# Rename 'userId' columns to avoid confusion
database = database.rename(columns={'userId': 'ml_userId'})
aux_data = aux_data.rename(columns={'userId': 'aux_userId'})

# Convert weights to a dictionary for fast lookup
movie_weights = dict(zip(weights_df['imdbId'].astype(str), weights_df['weight']))

# ---------------------- Similarity Measures ----------------------

def compute_rating_similarity(rating_db, rating_aux):
    """
    Compute similarity for ratings based on threshold.
    Returns 1 if |rating_db - rating_aux| <= 1, else 0.
    """
    if pd.isnull(rating_aux) or pd.isnull(rating_db):
        return 0
    return 1 if abs(rating_db - rating_aux) <= 1 else 0

def compute_date_similarity(epoch_db, epoch_aux, date_threshold=14 * 24 * 60 * 60):
    """
    Compute similarity for dates using epoch time.
    Returns 1 if |epoch_db - epoch_aux| <= date_threshold (default: 14 days in seconds), else 0.
    """
    if pd.isnull(epoch_aux) or pd.isnull(epoch_db):
        return 0
    delta = abs(epoch_db - epoch_aux)
    return 1 if delta <= date_threshold else 0

# ---------------------- Scoring Function ----------------------

def compute_score(aux_row, db_subset, movie_weights, rho0=RHO0, d0=D0):
    """
    Compute the Score(aux, r') for a given adversary record against a subset of database records.
    aux_row: Series from aux_data
    db_subset: DataFrame subset from database containing records that have movies in aux_row
    movie_weights: Dictionary mapping imdbId to weights
    rho0: Rating threshold parameter
    d0: Date threshold in seconds
    Returns: Series of scores indexed by database record index
    """
    scores = pd.Series(0, index=db_subset.index, dtype=float)
    
    for _, aux_record in aux_row.iterrows():
        aux_imdbId = aux_record['imdbId']
        aux_rating = aux_record['rating_value']
        aux_epoch = aux_record['review_date_epoch']
        
        # Find matching records in the database for this imdbId
        matching_db = db_subset[db_subset['imdbId'] == aux_imdbId]
        if matching_db.empty:
            continue
        
        # Compute similarities
        rating_sim = matching_db['rating'].apply(lambda x: compute_rating_similarity(x, aux_rating))
        date_sim = matching_db['timestamp'].apply(lambda x: compute_date_similarity(x, aux_epoch))
        
        # Compute the similarity component: exp((rho_i - rho'_i)/rho0) + exp((epoch_i - epoch'_i)/d0)
        exp_component = np.exp((aux_rating - matching_db['rating']) / rho0) + \
                        np.exp((aux_epoch - matching_db['timestamp']) / d0)
        
        # Apply weights
        weight = movie_weights.get(aux_imdbId, 1)
        weighted_exp = weight * exp_component
        
        # Update scores
        scores.loc[matching_db.index] += weighted_exp
    
    return scores

# ---------------------- Matching Criterion ----------------------

def matching_criterion(scores, phi=PHI):
    """
    Determine if there's a unique best match based on scores and eccentricity.
    scores: Series of similarity scores
    phi: Eccentricity parameter
    Returns: Index of best match or None
    """
    if scores.empty or scores.sum() == 0:
        return None
    
    max_score = scores.max()
    if len(scores) == 1:
        return scores.idxmax()
    
    # Get top two scores
    top_two = scores.nlargest(2)
    max_score = top_two.iloc[0]
    max2_score = top_two.iloc[1] if len(top_two) > 1 else 0
    
    sigma = scores.std()
    
    if sigma == 0:
        return None  # Avoid division by zero
    
    if (max_score - max2_score) / sigma < phi:
        return None
    else:
        return scores.idxmax()

# ---------------------- De-anonymization Function ----------------------

def de_anonymize_scoreboard_rh(database, aux_data, movie_weights, output_dir, exclude_top_n=None):
    """
    Perform de-anonymization using the Scoreboard-RH algorithm.
    database: DataFrame containing anonymized data
    aux_data: DataFrame containing adversary's auxiliary data
    movie_weights: Dictionary mapping imdbId to weights
    output_dir: Directory to save de-anonymization results
    exclude_top_n: Exclude top N most rated movies if specified
    """
    results = []
    
    # Optionally exclude top N movies based on rating counts
    if exclude_top_n:
        top_movies = database['imdbId'].value_counts().nlargest(exclude_top_n).index
        database_filtered = database[~database['imdbId'].isin(top_movies)]
    else:
        database_filtered = database.copy()
    
    # Iterate over each adversary record
    for idx, aux_row in tqdm(aux_data.iterrows(), total=aux_data.shape[0], desc="De-anonymizing"):
        # Find all database records that have at least one matching imdbId
        db_subset = database_filtered[database_filtered['imdbId'].isin(aux_row['imdbId'])]
        
        # Compute scores
        scores = compute_score(aux_row, db_subset, movie_weights)
        
        # Determine best match
        best_match_idx = matching_criterion(scores)
        
        if best_match_idx is not None:
            matched_record = database.loc[best_match_idx]
            results.append({
                'aux_userId': aux_row['aux_userId'],
                'matched_ml_userId': matched_record['ml_userId'],
                'imdbId': matched_record['imdbId'],
                'rating': matched_record['rating'],
                'timestamp': matched_record['timestamp'],
                'score': scores[best_match_idx]
            })
        else:
            results.append({
                'aux_userId': aux_row['aux_userId'],
                'matched_ml_userId': None,
                'imdbId': None,
                'rating': None,
                'timestamp': None,
                'score': None
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'deanonymization_results.csv'), index=False)
    print(f"De-anonymization results saved to {os.path.join(output_dir, 'deanonymization_results.csv')}")

# ---------------------- Execute De-anonymization ----------------------

print("Starting de-anonymization using Scoreboard-RH...")

# Example: De-anonymize with uniform settings
de_anonymize_scoreboard_rh(database, aux_data, movie_weights, OUTPUT_DIR_UNIFORM)

# Example: De-anonymize excluding top 100 most rated movies
de_anonymize_scoreboard_rh(database, aux_data, movie_weights, OUTPUT_DIR_NO_TOP_100, exclude_top_n=100)

# Example: De-anonymize excluding top 500 most rated movies
de_anonymize_scoreboard_rh(database, aux_data, movie_weights, OUTPUT_DIR_NO_TOP_500, exclude_top_n=500)

print("De-anonymization completed.")
